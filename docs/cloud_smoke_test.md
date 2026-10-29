# Cloud smoke test (Claude Code cloud session)

Date: 2026-10-29. Branch tested: `docs/lab-report-2-design` at `b457588`. No project code was changed.

## Result per step

| Step | Result | Notes |
|---|---|---|
| 1. Fetch, check out branch, data files present | PASS | `Unit_Cell_Catalog/Unit_Cell_Catalog.txt` (61.6 MB) and `data/samples/sample_lattices.txt` present |
| 2. Record machine | PASS | see below |
| 3a. Install with the documented CPU command (`--index-url https://download.pytorch.org/whl/cpu`) | **FAIL** | `download.pytorch.org` is blocked by this environment's network policy (proxy returns 403 to CONNECT). pip kept retrying for ~11 min without installing anything, so I killed it. |
| 3b. Fallback: `pip install -r requirements.txt` from PyPI only | PASS | 413 s (~7 min). Installs the CUDA build `torch 2.14.1+cu130` (runs fine on CPU, `cuda.is_available() == False`), but the venv is **5.5 GB** because of the NVIDIA wheels |
| 4. `check_data.py` on sample | PASS | 4 lattices, 0 problems in every category |
| 5. 5-epoch training on sample | PASS | 4.7 s wall time |
| 6. 3-epoch training on 2,000-lattice catalog subset, `--split group` | PASS | ~2.0 s/epoch, but this subset is not representative (see below) |
| 7. This report | PASS | |

## Machine

- CPU: 4 cores, Intel Xeon @ 2.80 GHz (`torch.get_num_threads() == 4`)
- RAM: 15 GB, no swap
- Disk: 30 GB free at start (session allowance, not the 252 GB `df` size)
- GPU: none (`nvidia-smi` not found)
- Python: **3.12.3** (`python3.12 -m venv .venv`; the default `python3` is 3.13.17)
- Versions installed: torch 2.14.1+cu130, torch_geometric 2.8.0.post1, numpy 2.5.3

## Step 4: `check_data.py data/samples/sample_lattices.txt`

```
Parsed 4 lattices
Z_avg mismatch: 0 | Poisson asymmetry: 0 | C not positive definite: 0 | No properties: 0
```

## Step 5: sample training (5 epochs)

Wall time 4.7 s (581,129 parameters). As the script warns, 4 lattices are reused for train/val/test, so this run only checks that the code works.

```
epoch 1  train loss 0.7818  val loss 0.6281
epoch 5  train loss 0.1931  val loss 0.2580
Test R²: Ex 0.794, Ey 0.989, Ez 0.697, Gyz 0.363, Gxz 0.741, Gxy 0.684, nu_yz 0.625, nu_xz 0.695, nu_xy 0.317
```

## Step 6: real catalog data

The catalog has 17,262 records. All of them parse; 40 have no properties, which leaves **17,222 complete records**. The subset files were written to `/tmp` and not committed.

### 6a. First 2,000 complete records (as requested)

`python train.py /tmp/subset.txt --split group --epochs 3 --out /tmp/subset.pt`
Split: 1,596 train / 204 val / 200 test.

- Parse + graph build (first run, cached afterwards): ~4 s
- **~2.0 s per epoch** (a 10-epoch rerun gave 2.1 s/epoch)
- Total wall time: 14 s with `--rebuild`

```
epoch 1  train loss 0.8225  val loss 0.5282
epoch 3  train loss 0.4534  val loss 0.3793
Test (200), loss 0.3676. R²: Ex 0.694, Ey 0.711, Ez 0.728, Gyz 0.622, Gxz 0.432, Gxy 0.602, nu_yz 0.373, nu_xz 0.633, nu_xy 0.459
```

### 6b. Why the first 2,000 are not representative

The catalog is ordered by cell size, which grows steadily through the file:

| Records | mean nodes (file) | max nodes | mean bars (file) | max bars |
|---|---|---|---|---|
| 0–2000 | 28 | 94 | 35 | 103 |
| 8000–10000 | 96 | 306 | 109 | 281 |
| 16000–17222 | 141 | 4,224 | 197 | 7,008 |

After graph building (deduplicated periodic graph), the first 2,000 average 11 nodes and 23 edges per lattice, while the full catalog averages 60 nodes and 105 edges. That is 8.6× the lattices but ~40× the edges. Extrapolating linearly from 6a would underestimate the cost by roughly 3.6×.

### 6c. Representative timing: 2,000 evenly spaced records

I made a second subset (every ~8.6th complete record, `/tmp/subset_strided.txt`) and ran 10 epochs with `--split group` (1,567 train / 218 val / 215 test):

- Graph build: ~95 s
- Epoch 1: ~8.5 s; epochs 2–10: **~7.5 s per epoch**
- 10-epoch test R²: Ex 0.626, Ey 0.743, Ez 0.697, Gyz 0.571, Gxz 0.555, Gxy 0.575, nu 0.46–0.51

### 6d. Full-catalog graph build (measured)

Parsing all 17,262 records takes 7 s. `build_periodic_graph` over all 17,222 complete records took **750 s (12.5 min)** on a single core, at 100% of one CPU and ~300 MB RAM. `load_dataset` caches the result in `data/processed/`, so this is a one-time cost per session. The container is ephemeral, though, so a fresh session pays it again unless the cache file is carried over.

## Estimate: 200 epochs on the full catalog on this machine

With the default 80/10/10 split there are ~13,780 training lattices, 8.8× the 6c subset:

- Per epoch: 7.5 s × 8.8 ≈ **~65 s/epoch**. This is linear in the number of lattices, using the representative size mix.
- 200 epochs: ≈ **3.6 h**, plus ~13 min one-time graph build, so **~4 h total**.
- Early stopping (`--patience 40`) may end the run sooner. The size-ordered tail, with cells up to 7,000 bars, can make some batches much slower, so treat ~4 h as a central estimate with roughly ±30% uncertainty.
- A naive extrapolation from the first-2,000 subset would have given ~18 s/epoch (~1 h), which is too optimistic.

For the Nov 1 demo, a representative (strided) subset of 2,000–5,000 lattices at 100–200 epochs fits in about 25 min to 2 h here.

## Issues found (not fixed; project code was not changed)

1. **The documented CPU install command cannot work in this cloud environment**, because `download.pytorch.org` is not on the network allowlist. Options: add that host to the environment's network policy (environment settings in Claude Code on the web: https://code.claude.com/docs/en/claude-code-on-the-web), or accept the PyPI install, which is CUDA-enabled, 5.5 GB, and takes ~7 min per fresh session.
2. **`check_data.py` reports a Z_avg mismatch on 1,885 of the first 2,000 catalog records.** Only the very simple cells match (e.g. 1 unique node). Example: record 6 in the subset has 26 nodes, which collapse to 17 unique, with 25 bars, giving graph Z 2.94 against the file's 4.00. The sample file has 0 mismatches, so the sample does not exercise this. It suggests that node deduplication or the periodic-edge handling in `lattice_gnn/graph.py` disagrees with how the catalog counts connectivity. Worth investigating before trusting results on the real catalog.
3. Every run prints `FutureWarning: torch.jit.script is deprecated` from inside torch. This is harmless.
