"""Check the baseline schedule and independent episode resets without Isaac Lab."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_control_module(filename):
    math = types.ModuleType("isaaclab.utils.math")
    math.quat_apply = lambda _quat, vector: vector
    math.quat_apply_inverse = lambda _quat, vector: vector
    math.compute_pose_error = lambda position, _rotation, target, _target_rotation, **_kwargs: (
        target - position, torch.zeros_like(position)
    )
    modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math,
    }
    path = Path(__file__).resolve().parents[1] / "scripts" / "franka_impedance" / "controllers" / filename
    spec = importlib.util.spec_from_file_location(f"{path.stem}_test_subject", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def fake_environment():
    count = 2
    body_pose = torch.zeros((count, 4, 7))
    body_pose[:, 1, :3] = torch.tensor([0.0, 0.0, 0.5])
    body_pose[:, 2, :3] = torch.tensor([0.0, -0.04, 0.45])
    body_pose[:, 3, :3] = torch.tensor([0.0, 0.04, 0.45])
    jacobian = torch.zeros((count, 1, 6, 7))
    jacobian[:, 0, 0, 0] = 1.0
    jacobian[:, 0, 1, 1] = 1.0
    jacobian[:, 0, 2, 2] = 1.0
    cube_pose = torch.zeros((count, 7))
    cube_pose[:, :3] = torch.tensor([0.0, 0.1, 0.3])
    robot_data = types.SimpleNamespace(
        body_link_pose_w=types.SimpleNamespace(torch=body_pose),
        body_link_jacobian_w=types.SimpleNamespace(torch=jacobian),
        joint_vel=types.SimpleNamespace(torch=torch.zeros((count, 7))),
    )
    robot = types.SimpleNamespace(data=robot_data, find_bodies=lambda *_args, **_kwargs: ([2, 3], None))
    cube = types.SimpleNamespace(data=types.SimpleNamespace(root_link_pose_w=types.SimpleNamespace(torch=cube_pose)))
    return types.SimpleNamespace(
        num_envs=count,
        device="cpu",
        actions=torch.zeros((count, 8)),
        robot=robot,
        observations=types.SimpleNamespace(ee_body_id=1),
        control=types.SimpleNamespace(arm_joint_ids=list(range(7)), limits=torch.full((count, 8), 100.0)),
        cfg=types.SimpleNamespace(arm_effort_scale=0.1),
        scene={"object": cube},
        episodes=types.SimpleNamespace(
            elapsed_steps=torch.zeros(count, dtype=torch.long),
            clearance=lambda: torch.tensor(0.0),
        ),
    )


class CartesianImpedanceTests(unittest.TestCase):
    def test_schedule_and_per_environment_reset(self):
        controller_type = load_control_module("cartesian_impedance.py").CartesianImpedanceGrasp
        env = fake_environment()
        controller = controller_type(env)
        self.assertEqual(controller.phase_name(35), "settle")
        self.assertEqual(controller.phase_name(36), "approach")
        self.assertEqual(controller.phase_name(236), "descend")
        self.assertEqual(controller.phase_name(386), "close")
        self.assertEqual(controller.phase_name(486), "lift")
        self.assertEqual(controller.phase_name(686), "hold")

        self.assertTrue(torch.equal(controller.act(), torch.zeros_like(env.actions)))
        env.episodes.elapsed_steps[:] = torch.tensor([36, 0])
        action = controller.act()
        self.assertTrue(torch.isfinite(action).all())
        self.assertGreater(action[0, :7].abs().sum().item(), 0.0)
        self.assertEqual(action[0, 7].item(), 1.0)
        self.assertTrue(torch.equal(action[1], torch.zeros(8)))
        planned_pinch = controller.pinch[0].clone()
        env.scene["object"].data.root_link_pose_w.torch[0, 0] += 1.0
        controller.act()
        self.assertTrue(torch.equal(controller.pinch[0], planned_pinch))

        env.episodes.elapsed_steps[:] = torch.tensor([386, 36])
        action = controller.act()
        self.assertEqual(action[0, 7].item(), -1.0)
        self.assertEqual(action[1, 7].item(), 1.0)
        controller.reset(torch.tensor([True, False]))
        self.assertTrue(torch.equal(controller.target[0], torch.zeros(3)))
        env.episodes.elapsed_steps[:] = torch.tensor([0, 486])
        action = controller.act()
        self.assertTrue(torch.equal(action[0], torch.zeros(8)))
        self.assertEqual(action[1, 7].item(), -1.0)

    def test_scripted_gripper_commands_match_current_mapping(self):
        scripted_actions = load_control_module("scripted_grasp.py").scripted_actions
        trajectory = scripted_actions(fake_environment())
        phase, _, action, _ = next(trajectory)
        self.assertEqual(phase, "approach")
        self.assertEqual(action[0, 7].item(), 1.0)
        for _ in range(349):
            next(trajectory)
        phase, _, action, _ = next(trajectory)
        self.assertEqual(phase, "close")
        self.assertEqual(action[0, 7].item(), -1.0)


if __name__ == "__main__":
    unittest.main()
