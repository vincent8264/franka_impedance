"""Check zero-residual equivalence and correction headroom."""

import importlib.util
import unittest
from pathlib import Path

import torch


path = Path(__file__).resolve().parents[1] / "scripts" / "franka_impedance" / "controllers" / "residual_effort.py"
spec = importlib.util.spec_from_file_location("residual_effort_test_subject", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResidualEffortTests(unittest.TestCase):
    def test_zero_residual_matches_clipped_baseline(self):
        baseline = torch.tensor([[1.4, -1.3, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
                                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        result = module.compose_residual_action(
            baseline, torch.zeros_like(baseline), arm_effort_scale=0.1,
            residual_arm_scale=0.02, residual_finger_effort_n=2.0,
        )
        torch.testing.assert_close(result, baseline.clamp(-1.0, 1.0))

    def test_residual_has_headroom_at_saturated_baseline(self):
        baseline = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
        residual = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
        result = module.compose_residual_action(
            baseline, residual, arm_effort_scale=0.1,
            residual_arm_scale=0.02, residual_finger_effort_n=2.0,
        )
        self.assertAlmostEqual(result[0, 0].item(), 1.2)
        self.assertAlmostEqual(result[0, 7].item() * module.CLOSING_EFFORT_N, -19.5, places=5)

    def test_finger_correction_can_cross_zero_effort(self):
        baseline = torch.zeros((1, 8))
        residual = torch.zeros_like(baseline)
        residual[0, 7] = 1.0
        result = module.compose_residual_action(
            baseline, residual, arm_effort_scale=0.1,
            residual_arm_scale=0.02, residual_finger_effort_n=2.0,
        )
        self.assertAlmostEqual(result[0, 7].item() * module.OPENING_EFFORT_N, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
