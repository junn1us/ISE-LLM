import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

metric_files = [OUT / f"deformation_metrics_1000iter_mf5_batch{b}_5fixtures.csv" for b in range(5, 10)]
pair_files = [OUT / f"deformation_pairwise_1000iter_mf5_batch{b}_5fixtures.csv" for b in range(5, 10)]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if not xs:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


all_rows = []
summary_rows = []

for metric_fp, pair_fp in zip(metric_files, pair_files):
    if not metric_fp.exists() or not pair_fp.exists():
        continue

    metrics = list(csv.DictReader(metric_fp.open()))
    pairs = list(csv.DictReader(pair_fp.open()))
    if not metrics:
        continue

    rmse = [float(r["rmse"]) for r in metrics]
    ret = [float(r["return_sum"]) for r in metrics]
    dmax = [float(r["dmax"]) for r in metrics]
    l2 = [float(r["l2_rms_diff"]) for r in pairs] if pairs else []

    batch_name = metric_fp.stem.replace("deformation_metrics_", "")
    summary_rows.append(
        {
            "batch": batch_name,
            "runs": len(metrics),
            "rmse_mean": mean(rmse),
            "rmse_std": std(rmse),
            "return_mean": mean(ret),
            "return_std": std(ret),
            "dmax_mean": mean(dmax),
            "pair_l2_mean": mean(l2) if l2 else float("nan"),
            "pair_l2_max": max(l2) if l2 else float("nan"),
        }
    )
    for r in metrics:
        rr = dict(r)
        rr["batch"] = batch_name
        all_rows.append(rr)

summary_csv = OUT / "deformation_summary_1000iter_mf5_batches5to9.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "batch",
            "runs",
            "rmse_mean",
            "rmse_std",
            "return_mean",
            "return_std",
            "dmax_mean",
            "pair_l2_mean",
            "pair_l2_max",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

report_txt = OUT / "deformation_summary_1000iter_mf5_batches5to9.txt"
with report_txt.open("w") as f:
    f.write("1000-iter MF5 batch summary (batches 5-9)\n")
    f.write("==========================================\n")
    for row in summary_rows:
        f.write(
            f"{row['batch']}: rmse_mean={row['rmse_mean']:.6f}, rmse_std={row['rmse_std']:.6f}, "
            f"return_mean={row['return_mean']:.6f}, pair_l2_mean={row['pair_l2_mean']:.6f}\n"
        )

    if all_rows:
        rmse_all = [float(r["rmse"]) for r in all_rows]
        ret_all = [float(r["return_sum"]) for r in all_rows]
        best = min(all_rows, key=lambda r: float(r["rmse"]))
        worst = max(all_rows, key=lambda r: float(r["rmse"]))
        f.write("\nGLOBAL\n")
        f.write(f"runs={len(all_rows)}\n")
        f.write(f"rmse_mean={mean(rmse_all):.6f}, rmse_std={std(rmse_all):.6f}\n")
        f.write(f"return_mean={mean(ret_all):.6f}, return_std={std(ret_all):.6f}\n")
        f.write(
            f"best_rmse={float(best['rmse']):.6f} ({best['checkpoint']}, batch={best['batch']})\n"
        )
        f.write(
            f"worst_rmse={float(worst['rmse']):.6f} ({worst['checkpoint']}, batch={worst['batch']})\n"
        )

print(f"Saved: {summary_csv}")
print(f"Saved: {report_txt}")
