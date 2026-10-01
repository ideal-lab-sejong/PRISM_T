# PRISM-T — reproducibility package

Code to reproduce the PRISM-T results of the paper: the main experiment and the
component / architecture ablations.

Datasets and results are not shipped — you create them: `data/Classical`
(47 ADBench `.npz` files, ~83 MB) and `results/`.

**Contents** — [1. Install](#1-install) · [2. Data](#2-data) ·
[3. Running the experiments](#3-running-the-experiments) ·
[4. Outputs](#4-outputs) · [5. Repository layout](#5-repository-layout)

## 1. Install

```bash
python -m venv venv && source venv/bin/activate    # Python >= 3.8
pip install -r requirements.txt
```

Dependencies are PyTorch (CPU is fine), NumPy, scikit-learn and pandas. No GPU is
required. There is no installation step for the package itself — run everything
**from the repository root**.

## 2. Data

```bash
python datasets/download_data.py     # fetches the 47 ADBench Classical .npz files into data/Classical/
```

Source: [ADBench](https://github.com/Minqi824/ADBench) (Han et al., NeurIPS
2022). Each `.npz` holds arrays `X` (N×d float) and `y` (N, 0=normal /
1=anomaly). Files already present in `data/Classical/` are skipped, so an
interrupted download resumes by re-issuing the same command.

Every run uses the semi-supervised split — train = 50% of the normals, test =
the held-out normals plus all anomalies — with a `StandardScaler` fit on the
training normals only; the metrics are AUROC and AUC-PR (average precision).

Two deliberate choices the reported numbers depend on: **no resampling** (no
upsampling below 1000 rows, no cap at 50000, unlike some ADBench baselines), and
a frozen BatchNorm with **`momentum=1.0`** — its statistics are estimated in one
pass over 4096-row chunks, each overwriting the last, so above 4096 training rows
only the final chunk's statistics are kept.

## 3. Running the experiments

Smoke test first — two datasets, one seed, a few seconds:

```bash
python scripts/run_prism_t.py --datasets 29_Pima 35_SpamBase --seeds 0
```

```
29_Pima              seed0  AUROC= 71.0  AUCPR= 70.2
35_SpamBase          seed0  AUROC= 85.7  AUCPR= 88.7
```

The main run is the same command with its defaults — the adopted configuration
(`--norm ibn --width 512 --layers 1 --activation leaky`, Mahalanobis shrinkage
λ = 10⁻³, a fixed absolute value) — over all 47 datasets and seeds 0-4:

```bash
python scripts/run_prism_t.py            # writes results/, prints the overall mean
```

Expected overall means (averaged over the 47 datasets): AUROC 86.4 / AUC-PR 65.9.

Both ablations run with one command — each arm is the main config with one
setting changed, and the script writes the comparison tables to
`results/ablation/comparison.md` once the arms finish. It reuses the main run
above as the reference row and skips any arm already computed:

```bash
python scripts/run_ablation_t.py
```

| Factor | Arms (main = `--norm ibn --width 512 --layers 1 --activation leaky`) |
|---|---|
| Component | `--norm none` (Base), `--norm instance` (+IN), `--norm batch` (+BN) |
| Architecture | `--width 128/256/1024`, `--layers 2/3`, `--activation relu/tanh/sigmoid` |

A single arm can also be run directly, e.g.
`python scripts/run_prism_t.py --norm none --out results/ablation/norm_none`;
`python scripts/run_prism_t.py --help` lists every flag.

## 4. Outputs

```
<out>/prism_t_results.csv       per (dataset, seed) row: AUROC, AUCPR, timing
<out>/prism_t_summary.csv       per-dataset mean±std over seeds
results/ablation/<arm>/...      the same two files, one directory per ablation arm
results/ablation/comparison.md  component + architecture comparison tables
```

`<out>` is `results/` by default (set per arm with `--out`). Each run also prints
the per-dataset summary and the overall mean AUROC / AUC-PR across all datasets.

## 5. Repository layout

```
datasets/    loader.py (semi-supervised split + StandardScaler),
             download_data.py (fetches the 47 ADBench Classical datasets)
models/      prism_t.py (the frozen random-projection network: PRISMT, IBN),
             mahalanobis.py (shrinkage Mahalanobis fit/score)
evaluation/  metrics.py (AUROC / AUC-PR, score sanitization)
scripts/     run_prism_t.py (the single experiment engine; defaults = the adopted
             main config), run_ablation_t.py (runs every ablation arm and writes
             the comparison tables)
data/        create it via datasets/download_data.py — datasets are not distributed
results/     created by scripts/run_prism_t.py
```

## License / citation

To be added upon publication. The 47 ADBench "Classical" datasets remain under
their own licences; this package redistributes neither the datasets nor derived
weights (there are none — nothing is trained).
