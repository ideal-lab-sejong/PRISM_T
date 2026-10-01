import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
RUN = THIS.parent / "run_prism_t.py"
MAIN = ROOT / "results"
ABL = ROOT / "results" / "ablation"

ARMS = [
    ("norm_none", ["--norm", "none"]),
    ("norm_in",   ["--norm", "instance"]),
    ("norm_bn",   ["--norm", "batch"]),
    ("w128",  ["--width", "128"]),
    ("w256",  ["--width", "256"]),
    ("w1024", ["--width", "1024"]),
    ("d2", ["--layers", "2"]),
    ("d3", ["--layers", "3"]),
    ("relu",    ["--activation", "relu"]),
    ("tanh",    ["--activation", "tanh"]),
    ("sigmoid", ["--activation", "sigmoid"]),
]


def run(out, flags, passthrough):
    if (Path(out) / "prism_t_summary.csv").exists():
        print(f"skip {Path(out).name} (summary exists)")
        return
    subprocess.run([sys.executable, str(RUN), *flags, "--out", str(out), *passthrough], check=True)


def overall(out):
    p = Path(out) / "prism_t_results.csv"
    if not p.exists():
        return None
    g = pd.read_csv(p).groupby("dataset")[["AUROC", "AUCPR"]].mean()
    return float(g.AUROC.mean()), float(g.AUCPR.mean())


def table(title, rows):
    lines = [f"### {title}", "| Config | AUROC | AUCPR |", "|---|---:|---:|"]
    for label, out in rows:
        v = overall(out)
        lines.append(f"| {label} | {v[0]:.2f} | {v[1]:.2f} |" if v else f"| {label} | - | - |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    passthrough = (["--seeds", *args.seeds] if args.seeds else []) + \
                  (["--datasets", *args.datasets] if args.datasets else [])

    if not (MAIN / "prism_t_summary.csv").exists():
        print("note: no main run in results/ yet — run `python scripts/run_prism_t.py` first "
              "for the reference rows; running the ablation arms anyway.")
    for name, flags in ARMS:                    # main is NOT run here; it is reused from results/
        run(ABL / name, flags, passthrough)

    tables = "\n\n".join([
        table("Component (normalization)",
              [("Base (none)", ABL / "norm_none"), ("+IN", ABL / "norm_in"),
               ("+BN", ABL / "norm_bn"), ("+IBN (main)", MAIN)]),
        table("Architecture — width",
              [("128", ABL / "w128"), ("256", ABL / "w256"),
               ("512 (main)", MAIN), ("1024", ABL / "w1024")]),
        table("Architecture — depth",
              [("1 (main)", MAIN), ("2", ABL / "d2"), ("3", ABL / "d3")]),
        table("Architecture — activation",
              [("leaky (main)", MAIN), ("relu", ABL / "relu"),
               ("tanh", ABL / "tanh"), ("sigmoid", ABL / "sigmoid")]),
    ])
    ABL.mkdir(parents=True, exist_ok=True)
    out = ABL / "comparison.md"
    out.write_text(tables + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
