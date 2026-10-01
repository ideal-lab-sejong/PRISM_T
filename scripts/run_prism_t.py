import sys, time, argparse
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from datasets.loader import load, list_datasets
from models.prism_t import PRISMT
from models.mahalanobis import fit_mahal, score_mahal
from evaluation.metrics import auroc, aucpr, sanitize_scores

ROOT = THIS.parent.parent

LAM, SCALE_AWARE = 1e-3, False


def prism_t(X_train, X_test, seed, width, layers, act, norm):
    t0 = time.perf_counter()
    net = PRISMT(X_train.shape[1], seed, width=width, layers=layers, act=act, norm=norm).fit(X_train)
    mu, prec = fit_mahal(net.features(X_train), lam_rel=LAM, scale_aware=SCALE_AWARE)
    t_fit = time.perf_counter() - t0
    t1 = time.perf_counter()
    s = score_mahal(net.features(X_test), mu, prec)
    return s, t_fit, time.perf_counter() - t1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", type=str, nargs="+", default=None, help="default: all found in data/Classical")
    ap.add_argument("--out", default=str(ROOT / "results"), help="output directory (main default: results/)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--activation", default="leaky", choices=["leaky", "relu", "tanh", "sigmoid"])
    ap.add_argument("--norm", default="ibn", choices=["none", "instance", "batch", "ibn"],
                     help="ibn = main (half instance norm + half BatchNorm); none/instance/batch "
                          "are the component-ablation arms.")
    args = ap.parse_args()

    datasets = args.datasets or list_datasets()
    if not datasets:
        raise SystemExit("No datasets found. Run `python datasets/download_data.py` first.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT = out_dir / "prism_t_results.csv"
    SUM = out_dir / "prism_t_summary.csv"
    rows = []
    for ds in datasets:
        for seed in args.seeds:
            Xtr, Xte, yte = load(ds, seed)
            s, t_fit, t_inf = prism_t(Xtr, Xte, seed, args.width, args.layers, args.activation, args.norm)
            s = sanitize_scores(s)
            roc = auroc(yte, s)
            pr = aucpr(yte, s)
            rows.append(dict(dataset=ds, seed=seed, AUROC=roc, AUCPR=pr,
                             n_test=len(yte), n_anomaly=int(yte.sum()),
                             fit_sec=t_fit, infer_sec=t_inf))
            print(f"{ds:20s} seed{seed}  AUROC={roc:5.1f}  AUCPR={pr:5.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df)} per-run rows -> {OUT}")

    g = df.groupby("dataset").agg(
        AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
        AUCPR_mean=("AUCPR", "mean"), AUCPR_std=("AUCPR", "std"),
        seeds=("seed", "nunique")).round(2).reset_index()
    g.to_csv(SUM, index=False)
    print("\n" + g.to_string(index=False))
    print(f"\n=== Overall (mean over {g.shape[0]} datasets):  "
          f"AUROC={g.AUROC_mean.mean():.2f}  AUCPR={g.AUCPR_mean.mean():.2f} ===")
    print(f"Saved per-dataset summary -> {SUM}")


if __name__ == "__main__":
    main()
