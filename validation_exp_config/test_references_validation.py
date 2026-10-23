import unittest
import numpy as np
from smartfixture_llm4teach.core.fixture_core import FixtureCore
from validate_physical_reference import compare_reference


class ReferenceHarnessTests(unittest.TestCase):
    def test_missing_external_provenance_rejected(self):
        with self.assertRaises(ValueError):
            compare_reference({})

    def test_synthetic_roundtrip_tests_harness_not_physics(self):
        config = dict(anchor_mode="stride", grid_size=9)
        core = FixtureCore(**config)
        core.reset()
        reference = {"source": "synthetic_unit_test_not_physical_evidence", "units": "mm",
            "boundary_conditions": "synthetic_free_surrogate", "environment": config,
            "coordinates_m": core.coords.tolist(), "cases": [{"case_id": "empty",
            "fixture_nodes": [], "displacement_y_mm": core.disp[:, 1].tolist()}]}
        result = compare_reference(reference)
        self.assertEqual(result["cases"][0]["field_rmse_mm"], 0)
        self.assertEqual(result["status"], "comparison_only_requires_engineering_review")

    def test_material_scaling_trend_not_fea_validation(self):
        thin = FixtureCore(sheet_thickness_m=0.002)
        thick = FixtureCore(sheet_thickness_m=0.004)
        np.testing.assert_allclose(thin._free_sag, 4 * thick._free_sag)


if __name__ == "__main__":
    unittest.main()
