"""Write every k-th complete record (one with properties) of a lattice file to a new file.

The catalog is sorted by cell size, so taking the first N would keep only small cells;
a stride spreads the subset over the whole catalog.

Usage:
    python scripts/make_subset.py Unit_Cell_Catalog/Unit_Cell_Catalog.txt data/raw/subset_2000.txt --n 2000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_gnn.parse import SEPARATOR, parse_record  # noqa: E402

LINE = "-" * 89


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--n", type=int, required=True, help="number of records to keep")
    args = parser.parse_args()

    blocks = []
    for block in SEPARATOR.split(Path(args.source).read_text(encoding="utf-8-sig")):
        if "Name:" not in block:
            continue
        try:
            if parse_record(block).props is not None:
                blocks.append(block.strip("\n"))
        except ValueError:
            pass
    stride = max(len(blocks) // args.n, 1)
    kept = blocks[::stride][:args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{LINE}\n{b}\n" for b in kept) + LINE + "\n", encoding="utf-8")
    print(f"{len(blocks)} complete records, every {stride}th -> {len(kept)} in {out}")


if __name__ == "__main__":
    main()
