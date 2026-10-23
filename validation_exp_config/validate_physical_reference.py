r"""Compare supplied, independently obtained displacement fields; never invent FEA evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from validation_runtime import sha256, write_json, source_hashes


def compare_reference(reference):
    if reference.get("units") != "mm" or not reference.get("source") or not reference.get("boundary_conditions"):
        raise ValueError("Reference requires source, boundary_conditions and units='mm'")
    core = FixtureCore(**reference["environment"])
    core.reset()
    if not np.allclose(reference["coordinates_m"], core.coords, atol=1e-8):
        raise ValueError("Reference node coordinates/order must match the core; interpolate externally and document it")
    rows = []
    for case in reference["cases"]:
        nodes = case["fixture_nodes"]
        if len(nodes) != len(set(nodes)) or any(type(n) is not int or n < 0 or n >= core.num_anchors for n in nodes):
            raise ValueError("Invalid reference fixture nodes")
        if len(nodes) < core.min_feedback_fixtures or (core.max_fixtures > 0 and len(nodes) > core.max_fixtures):
            raise ValueError("Reference layout outside configured fixture constraints")
        field = np.asarray(case["displacement_y_mm"], dtype=float)
        if field.shape != (len(core.coords),) or not np.isfinite(field).all():
            raise ValueError("Invalid reference field shape/values")
        core.mask[:] = 0
        core.mask[nodes] = 1
        core.sim_apply()
        predicted = core.disp[:, 1]
        rows.append({"case_id": case["case_id"], "fixture_nodes": nodes,
            "field_rmse_mm": float(np.sqrt(np.mean((predicted - field) ** 2))),
            "reference_peak_mm": float(np.max(np.abs(field))),
            "surrogate_peak_mm": core.sim_max_deformation()})
    if not rows:
        raise ValueError("Reference contains no cases")
    pairs, concordant = 0, 0
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            actual = a["reference_peak_mm"] - b["reference_peak_mm"]
            predicted = a["surrogate_peak_mm"] - b["surrogate_peak_mm"]
            if not np.isclose(actual, 0):
                pairs += 1
                concordant += int(actual * predicted > 0)
    return {"source": reference["source"], "boundary_conditions": reference["boundary_conditions"],
        "source_hashes": source_hashes(), "cases": rows,
        "layout_ranking_agreement": concordant / pairs if pairs else None,
        "non_tied_reference_pairs": pairs,
        "status": "comparison_only_requires_engineering_review",
        "note": "Input provenance is user-supplied, not independently authenticated. No automatic physical-validity claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    result = compare_reference(json.loads(Path(args.reference).read_text()))
    result["reference_sha256"] = sha256(args.reference)
    write_json(args.out, result)
