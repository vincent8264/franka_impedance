"""Assertions and deterministic fixtures for the Newton smoke tests."""

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_newton.sensors.contact_sensor import ContactSensorCfg

from direct_rl_env import FINGER_PARENT, ContactGraspSceneCfg
from grasp_scene_cfg import TABLE_SIZE_M


@configclass
class ContactProbeSceneCfg(ContactGraspSceneCfg):
    left_table = ContactSensorCfg(
        prim_path=f"{FINGER_PARENT}/panda_leftfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )
    right_table = ContactSensorCfg(
        prim_path=f"{FINGER_PARENT}/panda_rightfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )


def move_asset(asset, ids, pose):
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(pose[:, :6]), env_ids=ids)


def check_rollout(env):
    action = torch.zeros_like(env.actions)
    assert action.shape == (env.num_envs, 8)
    initial, _ = env.reset()
    initial = initial["policy"].clone()
    assert initial.shape == (env.num_envs, 46) and torch.isfinite(initial).all()

    def step(actions):
        assert torch.isfinite(actions).all()
        obs, rewards, success, timeout, _ = env.step(actions)
        assert obs["policy"].shape == initial.shape and torch.isfinite(obs["policy"]).all()
        assert rewards.shape == (env.num_envs,) and torch.isfinite(rewards).all()
        return obs["policy"], success, timeout

    drift = {}
    for compensate in (True, False):
        env.cfg.gravity_compensation = compensate
        env.reset()
        for _ in range(24):
            obs, success, timeout = step(action)
            assert not (success | timeout).any()
        drift[compensate] = (obs[:, :7] - initial[:, :7]).abs().max().item()
    env.cfg.gravity_compensation = True
    env.reset()
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    finger_joints = env.observations.joint_ids[7:9]
    env.robot.write_joint_position_to_sim_index(
        position=action.new_full((env.num_envs, 2), 0.02), env_ids=ids, joint_ids=finger_joints
    )
    finger_ids = env.robot.find_bodies(["panda_leftfinger", "panda_rightfinger"])[0]

    def separation():
        pos = env.robot.data.body_link_pose_w.torch[:2, finger_ids, :3]
        return torch.linalg.vector_norm(pos[:, 0] - pos[:, 1], dim=-1)

    before = separation()
    action[0, 7], action[1, 7] = 0.1, -0.1
    for _ in range(3):
        obs, _, _ = step(action)
    after = separation()
    print(f"Gripper separation before: {before.tolist()} m, after (+/-): {after.tolist()} m")
    assert (obs[:2, 18] > 0).all() and abs(after[0] - after[1]) > 0.0005, obs[:2, 18]
    sign = "positive" if after[0] > after[1] else "negative"
    print(f"Opening direction: {sign} effort")
    print(f"Zero-action maximum arm drift over 0.2 s: {drift}")

    env.reset()
    for _ in range(16):
        _, success, timeout = step(2.0 * torch.rand_like(action) - 1.0)
        assert not (success | timeout).any()

    env.reset()
    action.zero_()
    ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    cube = env.scene["object"]
    raised = cube.data.default_root_pose.torch[:1].clone()
    raised[:, :3] += env.scene.env_origins[:1]
    raised[:, 2] += 0.25
    # Scripted height fixture exercises the exact success path without a trained policy.
    for index in range(env.episodes.hold_steps):
        move_asset(cube, ids, raised)
        obs, success, timeout = step(action)
        assert bool(success[0]) == (index + 1 == env.episodes.hold_steps)
        assert not success[1:].any() and not timeout.any()
    assert env.episode_length_buf[0] == 0 and env.episode_length_buf[1] == env.episodes.hold_steps
    assert torch.allclose(obs[0, 33:36], initial[0, 33:36], atol=1e-4)
    for _ in range(env.episodes.max_steps - env.episodes.hold_steps):
        obs, success, timeout = step(action)
        assert not success.any()
    assert not timeout[0] and timeout[1:].all()
    assert (env.episode_length_buf[1:] == 0).all()
    assert torch.allclose(obs[1:, 33:36], initial[1:, 33:36], atol=1e-4)
    print(f"Rollout passed: success after {env.episodes.hold_steps} ticks, timeout after {env.episodes.max_steps} ticks")


def check_contacts(env):
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    action = torch.zeros_like(env.actions)
    robot, cube, table = (env.scene[name] for name in ("robot", "object", "table"))
    left = robot.find_bodies("panda_leftfinger")[0][0]
    right = robot.find_bodies("panda_rightfinger")[0][0]
    cube_sensors = env.cube_sensors
    table_sensors = (env.scene["left_table"], env.scene["right_table"])

    def forces():
        result = torch.stack([env.contact_force(sensor) for sensor in (*cube_sensors, *table_sensors)], dim=-1)
        assert torch.isfinite(result).all()
        return result

    env.reset()
    env.step(action)
    no_contact = forces()
    assert (no_contact < 0.05).all(), no_contact

    env.reset()
    finger = robot.data.body_link_pose_w.torch[:, [left, right]].clone()
    hand_quat = robot.data.body_link_pose_w.torch[:, env.observations.ee_body_id, 3:7].clone()
    pinch_center = finger[:, :, :3].mean(dim=1)
    offset = torch.zeros_like(pinch_center)
    offset[:, 2] = 0.02
    tip_center = pinch_center + quat_apply(hand_quat, offset)
    cube_pose = cube.data.default_root_pose.torch.clone()
    cube_pose[:, :3] = tip_center
    cube_pose[:, 3:7] = hand_quat
    move_asset(cube, ids, cube_pose)
    action[:, 7] = -0.5
    cube_peak = torch.zeros_like(no_contact)
    bilateral_cube = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for _ in range(24):
        env.step(action)
        measured = forces()
        bilateral_cube |= (measured[:, :2] > 0.5).all(dim=-1)
        cube_peak = torch.maximum(cube_peak, measured)

    env.reset()
    table_pose = table.data.root_link_pose_w.torch.clone()
    table_pose[:, 2] = tip_center[:, 2] - TABLE_SIZE_M[2] * 0.5 + 0.008
    move_asset(table, ids, table_pose)
    action.zero_()
    env.step(action)
    table_only = forces()
    print(f"No-contact forces: {no_contact.tolist()}")
    print(f"Cube fixture peak forces: {cube_peak.tolist()}")
    print(f"Table fixture forces: {table_only.tolist()}")
    assert bilateral_cube.all() and (cube_peak[:, 2:] < 0.05).all(), cube_peak
    assert (table_only[:, :2] < 0.05).all() and (table_only[:, 2:] > 0.5).all(), table_only
    print("Newton distinguished cube contact, table contact, and no contact on both fingers.")
