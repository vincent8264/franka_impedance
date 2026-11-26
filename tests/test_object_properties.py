"""Check paired property draws without launching the simulator."""

import argparse
import importlib.util
import types
import unittest
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "franka_impedance" / "envs" / "object_properties.py"
spec = importlib.util.spec_from_file_location("object_properties_test_subject", MODULE)
properties = importlib.util.module_from_spec(spec)
spec.loader.exec_module(properties)


class ObjectPropertyTests(unittest.TestCase):
    def test_cli_defaults_and_nominal_override(self):
        parser = argparse.ArgumentParser()
        properties.add_property_args(parser)
        defaults = parser.parse_args([])
        self.assertEqual(properties.parse_property_ranges(parser, defaults), ((0.1, 0.5), (0.2, 0.8)))
        nominal = parser.parse_args(["--mass-range", "0.2", "0.2", "--friction-range", "0.5", "0.5"])
        self.assertEqual(properties.parse_property_ranges(parser, nominal), ((0.2, 0.2), (0.5, 0.5)))

    def test_draws_are_paired_across_reset_order(self):
        ids = torch.tensor([0, 2, 5])
        episodes = torch.tensor([0, 3, 1])
        ranges = ((0.1, 0.4), (0.3, 0.8))
        mass, friction = properties.sample_properties(42, ids, episodes, *ranges)
        reverse_mass, reverse_friction = properties.sample_properties(42, ids.flip(0), episodes.flip(0), *ranges)
        torch.testing.assert_close(mass, reverse_mass.flip(0), rtol=0, atol=0)
        torch.testing.assert_close(friction, reverse_friction.flip(0), rtol=0, atol=0)
        self.assertTrue(((mass >= 0.1) & (mass <= 0.4)).all())
        self.assertTrue(((friction >= 0.3) & (friction <= 0.8)).all())
        self.assertFalse(torch.equal(mass, properties.sample_properties(42, ids, episodes + 1, *ranges)[0]))

    def test_nominal_and_invalid_ranges(self):
        ids = torch.tensor([0, 1])
        episodes = torch.zeros(2, dtype=torch.long)
        mass, friction = properties.sample_properties(42, ids, episodes, (0.2, 0.2), (0.5, 0.5))
        torch.testing.assert_close(mass, torch.full((2,), 0.2), rtol=0, atol=0)
        torch.testing.assert_close(friction, torch.full((2,), 0.5), rtol=0, atol=0)
        for bounds in ((0.0, 0.2), (0.4, 0.3), (float("nan"), 0.5)):
            with self.assertRaises(ValueError):
                properties.validate_range("mass", bounds)

    def test_completed_episode_keeps_its_pre_reset_properties(self):
        live = types.SimpleNamespace(
            mass_kg=torch.tensor([0.2, 0.3]), friction=torch.tensor([0.5, 0.6])
        )
        env = types.SimpleNamespace(
            num_envs=2, device="cpu", episodes=types.SimpleNamespace(properties=live)
        )
        record = properties.EpisodePropertyRecord(env)
        live.mass_kg[0] = 0.4
        live.friction[0] = 0.7
        record.record(torch.tensor([True, False]), torch.tensor([0, 0]))
        self.assertEqual(record.samples, [{
            "environment": 0, "episode": 0, "mass_kg": float(torch.tensor(0.2)),
            "friction": 0.5,
        }])
        self.assertEqual(record.mass_kg[0], live.mass_kg[0])
        self.assertEqual(record.friction[0], live.friction[0])


if __name__ == "__main__":
    unittest.main()
