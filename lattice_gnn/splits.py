"""Train / val / test splits, saved to JSON so every run uses the same one.

A split file lists *family* names (see `family`) for each part, so it can be made once
from the full catalog and applied to any subset of it: a lattice's part depends only on
its family, never on which other lattices happen to be in the file.

The locked split (frozen 2026-10-30; never regenerate or edit it) was made with:
    python -m lattice_gnn.splits Unit_Cell_Catalog/Unit_Cell_Catalog.txt --out splits/main.json \
        --max-family 200 --stratify
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

NAME = re.compile(r"^Name:\s*(\S+)", re.MULTILINE)
OTHER = re.compile(r"^Name:\s*(\S+)\s*\nOther name\(s\):\s*(.*)$", re.MULTILINE)
PARTS = ("train", "val", "test")


def family(name: str) -> str:
    """Group key: same symmetry and connectivity, e.g. cub_Z08.0_E3 or cub_Z08.0_R12 -> cub_Z08.0.

    _E<n> names come from EPINET and _R<n> from RCSR.
    """
    return re.sub(r"_[ER]\d+$", "", name)


def read_names(path: str | Path) -> list[str]:
    """Lattice names in a data file, without parsing the rest of each record."""
    return NAME.findall(Path(path).read_text(encoding="utf-8-sig"))


def read_duplicates(path: str | Path) -> list[tuple[str, str]]:
    """(name, other name) pairs that are both records in the file: the same structure,
    listed once from each database. A few pairs have different Z in their names, so they
    fall in different families; family_split keeps such families in the same part.
    """
    records = OTHER.findall(Path(path).read_text(encoding="utf-8-sig"))
    names = {name for name, _ in records}
    return [(name, other.strip()) for name, others in records
            for other in others.split(",") if other.strip() in names and other.strip() != name]


def crystal_system(name: str) -> str:
    """cub, tet, ort, hex, trig or mon: the prefix of every catalog name."""
    return name.split("_")[0]


def family_split(names, val_frac: float, test_frac: float, seed: int, duplicates=(),
                 max_family: int | None = None, stratify: bool = False) -> dict:
    """Whole families go to test, then val, until each holds its fraction of the lattices.

    Families linked by a duplicate pair (see read_duplicates) move together.
    max_family: families with more lattices than this always go to train, so no single
        family can make up most of val or test.
    stratify: fill test and val separately within each crystal system, each to its share
        of that system's lattices, so every system is represented in proportion.
    """
    root = {}

    def find(f):
        while root.setdefault(f, f) != f:
            f = root[f]
        return f

    for a, b in duplicates:
        root[find(family(a))] = find(family(b))
    groups = {}
    for name in names:
        groups.setdefault(find(family(name)), []).append(name)
    strata = {}   # stratum -> group keys; one stratum ("all") unless stratify
    for key in sorted(groups):
        strata.setdefault(crystal_system(key) if stratify else "all", []).append(key)
    rng = random.Random(seed)
    split = {part: [] for part in PARTS}
    counts = dict.fromkeys(PARTS, 0)
    for stratum in sorted(strata):
        keys = strata[stratum]
        total = sum(len(groups[k]) for k in keys)
        rng.shuffle(keys)
        filled = dict.fromkeys(PARTS, 0)
        for key in keys:
            if max_family is not None and len(groups[key]) > max_family:
                part = "train"
            elif filled["test"] < test_frac * total:
                part = "test"
            elif filled["val"] < val_frac * total:
                part = "val"
            else:
                part = "train"
            split[part] += {family(n) for n in groups[key]}
            filled[part] += len(groups[key])
            counts[part] += len(groups[key])
    return {"by": "family", **{part: sorted(split[part]) for part in PARTS},
            "meta": {"val_frac": val_frac, "test_frac": test_frac, "seed": seed,
                     "max_family": max_family, "stratify": stratify, "lattices": counts}}


def random_split(names, val_frac: float, test_frac: float, seed: int) -> dict:
    """Each lattice goes to a part at random (families leak across parts)."""
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n_test, n_val = round(test_frac * len(names)), round(val_frac * len(names))
    return {"by": "name", "train": sorted(names[n_test + n_val:]),
            "val": sorted(names[n_test:n_test + n_val]), "test": sorted(names[:n_test]),
            "meta": {"val_frac": val_frac, "test_frac": test_frac, "seed": seed}}


def save_split(split: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))


def load_split(path: str | Path) -> dict:
    split = json.loads(Path(path).read_text())
    if split.get("by") not in ("family", "name"):
        raise ValueError(f"{path}: 'by' must be 'family' or 'name'")
    return split


def apply_split(graphs, split: dict) -> tuple[list, list, list]:
    """Sort graphs into (train, val, test). Every graph must be covered by the split."""
    key = family if split["by"] == "family" else (lambda name: name)
    part_of = {k: part for part in PARTS for k in split[part]}
    parts = {part: [] for part in PARTS}
    missing = []
    for g in graphs:
        part = part_of.get(key(g.name))
        if part is None:
            missing.append(g.name)
        else:
            parts[part].append(g)
    if missing:
        raise ValueError(f"{len(missing)} lattices are not in the split (e.g. {missing[:3]}); "
                         "make the split from a file that contains them")
    return parts["train"], parts["val"], parts["test"]


def main():
    parser = argparse.ArgumentParser(description="Make a train/val/test split file.")
    parser.add_argument("paths", nargs="+", help="lattice data file(s); use the full catalog")
    parser.add_argument("--out", required=True, help="e.g. splits/main.json")
    parser.add_argument("--by", choices=["family", "name"], default="family")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-family", type=int, default=None,
                        help="families with more lattices than this always go to train")
    parser.add_argument("--stratify", action="store_true",
                        help="fill val and test within each crystal system")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; splits are frozen once made (use --force to replace)")
    names = [n for path in args.paths for n in read_names(path)]
    if args.by == "family":
        duplicates = [pair for path in args.paths for pair in read_duplicates(path)]
        split = family_split(names, args.val_frac, args.test_frac, args.seed, duplicates,
                             args.max_family, args.stratify)
    else:
        split = random_split(names, args.val_frac, args.test_frac, args.seed)
    split["meta"]["source"] = [str(p) for p in args.paths]
    save_split(split, out)
    print(f"{len(names)} lattices -> {out}: " +
          ", ".join(f"{len(split[p])} {p} {'families' if args.by == 'family' else 'lattices'}"
                    for p in PARTS))


if __name__ == "__main__":
    main()
