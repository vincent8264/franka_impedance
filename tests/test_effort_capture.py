"""Verify that validation records the applied tensor and restores recorder state."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from franka_impedance.diagnostics.effort_capture import AppliedEffortCapture


class AppliedEffortCaptureTests(unittest.TestCase):
    def test_capture_is_pre_reset_copy(self):
        previous = object()
        env = SimpleNamespace(step_recorder=previous)
        applied = torch.tensor([[2.0, -3.0]])
        with AppliedEffortCapture(env) as capture:
            self.assertIs(env.step_recorder, capture)
            capture.capture_effort(torch.zeros_like(applied), applied)
            applied.zero_()
            self.assertEqual(capture.applied.tolist(), [[2.0, -3.0]])
        self.assertIs(env.step_recorder, previous)


if __name__ == "__main__":
    unittest.main()
