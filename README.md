# How Far Can Anomaly Detection Go Before Learning? Random Neural Features from Tables to Pixels

**PRISM FrameWork**

![Architecture](./images/PRISM_Framework.png)

Authors: Hyojun Yi, Geunyoung Kim, Hogeon Seo, and Seunghyoung Ryu

paper: 


## Introduction

This repository contains the official implementation of **PRISM-T**, the tabular anomaly detection module of PRISM. PRISM-T investigates how effectively a neural network with randomly initialized, frozen weights can detect anomalies in tabular data without any training.

A single PRISM block — a frozen random linear layer (width 512), an IBN layer (half instance normalization, half BatchNorm) and a LeakyReLU — maps each sample to a descriptor, which is scored by a global Mahalanobis distance to the normal reference set. No weights are optimized.


## Installation

```bash
python -m venv venv && source venv/bin/activate    # Python >= 3.8
pip install -r requirements.txt
```

Dependencies are PyTorch (CPU is fine), NumPy, scikit-learn and pandas. No GPU is required. Run everything from the repository root.


## Data

PRISM-T is evaluated on the 47 **ADBench Classical** datasets ([Han et al., NeurIPS 2022](https://github.com/Minqi824/ADBench)). Download them by running:

```bash
python datasets/download_data.py
```

Files already present are skipped, so an interrupted download resumes by re-running the same command. After preparation, the datasets will be available at:

```text
data/
└── Classical/    # 47 .npz files (X: N×d features, y: 0 = normal / 1 = anomaly)
```

Every run uses the semi-supervised split: 50% of the normal samples form the reference (training) set, and the held-out normals plus all anomalies form the test set. A `StandardScaler` is fit on the reference set only.


## Running the experiments

The driver below does not download anything, so `data/Classical` must already exist.

```bash
bash scripts/run_main.sh
```

This runs the adopted configuration on all 47 datasets with seeds 0–4 and reports AUROC and AUC-PR. Expected overall means: **AUROC 86.4 / AUC-PR 65.9**.

For a quick check on a subset:

```bash
python scripts/run_prism_t.py --datasets 29_Pima 35_SpamBase --seeds 0
```

`python scripts/run_prism_t.py --help` lists every flag.


## Repository layout

```
models/      prism_t.py (the PRISM-T encoder: PRISMT, IBN),
             mahalanobis.py (shrinkage Mahalanobis fit/score)
evaluation/  metrics.py (AUROC/AUC-PR, score sanitization)
datasets/    loader.py (semi-supervised split + StandardScaler),
             download_data.py (fetches the 47 ADBench Classical datasets)
scripts/     run_prism_t.py (the single experiment engine; defaults = the
             adopted main config), run_main.sh (main experiment)
data/        create it — datasets are not distributed
results/     created by the run scripts: main/
```


## Acknowledgement

This work was partially supported in part by the IITP (Institute of Information & Communications Technology Planning & Evaluation) - ICAN (ICT Challenge and Advanced Network of HRD) grant funded by the Korea government (Ministry of Science and ICT) (IITP-2026-RS-2024-00436528), and in part by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2024-00454514)


## License / citation

To be added upon publication. The 47 ADBench "Classical" datasets remain under
their own licences; this package redistributes neither the datasets nor derived
weights (there are none — nothing is trained).
