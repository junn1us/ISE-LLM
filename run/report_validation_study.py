"""Render tables and figures from one verified set of evaluation records."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from validation_runtime import sha256, write_json


def report(study):
    root = Path(study)
    manifest = json.loads((root / "study_manifest.json").read_text())
    if (manifest["status"] != "completed"
            or manifest["evaluation_records_sha256"] != sha256(root / "evaluation_records.json")
            or manifest["validation_curves_sha256"] != sha256(root / "validation_curves.json")):
        raise ValueError("Incomplete study or changed evaluation records")
    records = json.loads((root / "evaluation_records.json").read_text())
    curves = json.loads((root / "validation_curves.json").read_text())
    rows = []
    for split in ("validation", "test", "ood"):
        for condition in sorted({r["condition"] for r in records}):
            for phase in ("initial", "final"):
                selected = [r for r in records if r["split"] == split and r["condition"] == condition and r["phase"] == phase]
                if not selected:
                    continue
                means = [float(np.mean([r["normalized_max_deformation"] for r in selected if r["training_seed"] == seed]))
                         for seed in sorted({r["training_seed"] for r in selected})]
                rows.append({"split": split, "condition": condition, "phase": phase,
                    "training_seeds": len(means), "mean_normalized_max_deformation": float(np.mean(means)),
                    "std_across_training_seed_means": float(np.std(means, ddof=1)) if len(means) > 1 else None})
    with (root / "summary_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for condition in sorted({r["condition"] for r in curves}):
        points = sorted({r["ppo_transitions"] for r in curves if r["condition"] == condition})
        values = [[r["mean_normalized_max_deformation"] for r in curves
                   if r["condition"] == condition and r["ppo_transitions"] == point] for point in points]
        ax.plot(points, [np.mean(v) for v in values], marker="o", label=condition)
    ax.set(xlabel="PPO training transitions (pretraining cost reported separately)",
           ylabel="Validation peak deformation / free-sag peak",
           title=f"Synthetic validation — {manifest['scientific_status']}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(root / "validation_learning_curve.png", dpi=160)
    plt.close(fig)
    write_json(root / "report_provenance.json", {
        "evaluation_records_sha256": sha256(root / "evaluation_records.json"),
        "validation_curves_sha256": sha256(root / "validation_curves.json"),
        "summary_table_sha256": sha256(root / "summary_table.csv"),
        "figure_sha256": sha256(root / "validation_learning_curve.png"),
        "uncertainty": "Table SD across training-seed means, not a confidence interval; figure shows means only."})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    report(parser.parse_args().study)
