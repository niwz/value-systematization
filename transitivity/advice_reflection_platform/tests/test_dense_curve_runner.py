from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from advice_reflection_platform.scripts.run_dense_curve_pilot import (
    _load_reusable_prior_artifacts,
    _repeat_indices,
)


class DenseCurveRunnerTests(unittest.TestCase):
    def test_repeat_indices_can_start_after_existing_repeats(self) -> None:
        self.assertEqual(list(_repeat_indices(repeats_per_order=5, repeat_start_index=6)), [6, 7, 8, 9, 10])

    def test_reused_prior_artifacts_skip_baseline_and_filter_requested_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            analysis_path = Path(tmp_dir) / "analysis.json"
            reflection_artifact = {
                "condition_name": "reflection",
                "family_key": "family_x",
                "prompt": "Turn 1 reflection prompt",
                "prior_text": "Turn 1 reflection answer",
            }
            analysis_path.write_text(
                json.dumps(
                    {
                        "family_key": "family_x",
                        "prior_artifacts": {
                            "reflection": reflection_artifact,
                            "placebo": {
                                "condition_name": "placebo",
                                "family_key": "family_x",
                                "prompt": "Turn 1 placebo prompt",
                                "prior_text": "Turn 1 placebo answer",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            artifacts = _load_reusable_prior_artifacts(
                analysis_path,
                ["baseline", "reflection"],
                family_key="family_x",
            )

        self.assertEqual(artifacts, {"reflection": reflection_artifact})

    def test_reused_prior_artifacts_fail_when_requested_condition_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            analysis_path = Path(tmp_dir) / "analysis.json"
            analysis_path.write_text(
                json.dumps(
                    {
                        "family_key": "family_x",
                        "prior_artifacts": {
                            "reflection": {
                                "condition_name": "reflection",
                                "family_key": "family_x",
                                "prompt": "Turn 1 reflection prompt",
                                "prior_text": "Turn 1 reflection answer",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "placebo"):
                _load_reusable_prior_artifacts(
                    analysis_path,
                    ["baseline", "reflection", "placebo"],
                    family_key="family_x",
                )


if __name__ == "__main__":
    unittest.main()
