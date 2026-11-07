"""Shared observations, episode resets, and lift-success detection."""

import math

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

from grasp_scene_cfg import OBJECT_SIZE_M, TABLE_SIZE_M


OBSERVATION_FIELDS = (
    "joint_pos",  # seven arm joints, then the two finger joints; radians/meters
    "joint_vel",  # matching joint velocities
    "gripper_width",  # sum of both finger positions, meters
    "gripper_speed",  # sum of both finger velocities, meters/second
    "ee_pose_b",  # panda_hand position + xyzw quaternion in robot-base frame
    "ee_vel_b",  # panda_hand linear + angular velocity in robot-base frame
    "object_pose_b",  # cuboid position + xyzw quaternion in robot-base frame
    "object_vel_b",  # cuboid linear + angular velocity in robot-base frame
)
OBSERVATION_DIM = 46


class GraspObservations:
    """Read the same state vector for classical and learned controllers."""

    def __init__(self, scene):
        self.robot = scene["robot"]
        self.object = scene["object"]
        joint_names = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
        self.joint_ids, found_names = self.robot.find_joints(joint_names, preserve_order=True)
        if found_names != joint_names:
            raise RuntimeError(f"Expected observation joints {joint_names}, got {found_names}")
        self.ee_body_id = self.robot.find_bodies("panda_hand")[0][0]

    def read(self) -> dict[str, torch.Tensor]:
        """Return named fields and their concatenated (num_envs, 46) policy vector."""
        robot = self.robot
        base_pose_w = robot.data.root_link_pose_w.torch
        base_vel_w = robot.data.root_link_vel_w.torch
        ee_pose_w = robot.data.body_link_pose_w.torch[:, self.ee_body_id]
        ee_vel_w = robot.data.body_link_vel_w.torch[:, self.ee_body_id]
        object_pose_w = self.object.data.root_link_pose_w.torch
        object_vel_w = self.object.data.root_link_vel_w.torch

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pose_w[:, :3], base_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )
        object_pos_b, object_quat_b = subtract_frame_transforms(
            base_pose_w[:, :3], base_pose_w[:, 3:7], object_pose_w[:, :3], object_pose_w[:, 3:7]
        )

        def relative_velocity(pose_w: torch.Tensor, vel_w: torch.Tensor) -> torch.Tensor:
            offset_w = pose_w[:, :3] - base_pose_w[:, :3]
            linear_w = vel_w[:, :3] - base_vel_w[:, :3] - torch.cross(base_vel_w[:, 3:], offset_w, dim=-1)
            angular_w = vel_w[:, 3:] - base_vel_w[:, 3:]
            return torch.cat(
                (quat_apply_inverse(base_pose_w[:, 3:7], linear_w), quat_apply_inverse(base_pose_w[:, 3:7], angular_w)),
                dim=-1,
            )

        joint_pos = robot.data.joint_pos.torch[:, self.joint_ids].clone()
        joint_vel = robot.data.joint_vel.torch[:, self.joint_ids].clone()
        fields = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "gripper_width": joint_pos[:, 7:9].sum(dim=-1, keepdim=True),
            "gripper_speed": joint_vel[:, 7:9].sum(dim=-1, keepdim=True),
            "ee_pose_b": torch.cat((ee_pos_b, ee_quat_b), dim=-1),
            "ee_vel_b": relative_velocity(ee_pose_w, ee_vel_w),
            "object_pose_b": torch.cat((object_pos_b, object_quat_b), dim=-1),
            "object_vel_b": relative_velocity(object_pose_w, object_vel_w),
        }
        fields["policy"] = torch.cat([fields[name] for name in OBSERVATION_FIELDS], dim=-1)
        return fields


class GraspEpisodes:
    """Fixed-length episodes with a continuous table-clearance success window."""

    def __init__(self, scene, physics_dt: float, episode_seconds: float = 12.0):
        if physics_dt <= 0 or episode_seconds <= 0:
            raise ValueError("Physics timestep and episode length must be positive")
        self.scene = scene
        self.robot = scene["robot"]
        self.object = scene["object"]
        self.table = scene["table"]
        self.physics_dt = physics_dt
        self.max_steps = math.ceil(episode_seconds / physics_dt)
        self.hold_steps = math.ceil(0.5 / physics_dt)
        self.elapsed_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=scene.device)
        self.above_steps = torch.zeros_like(self.elapsed_steps)
        self.episode_number = torch.zeros_like(self.elapsed_steps)

    def reset(self, env_ids: list[int] | None = None) -> None:
        """Restore the default robot and cuboid state for selected environments."""
        if env_ids is None:
            env_ids = list(range(self.scene.num_envs))
        if not env_ids:
            return

        robot_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        robot_pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=self.robot.data.default_joint_pos.torch[env_ids].clone(), env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.robot.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids
        )

        object_pose = self.object.data.default_root_pose.torch[env_ids].clone()
        object_pose[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(
            root_velocity=self.object.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
        )
        self.scene.reset(env_ids)
        self.elapsed_steps[env_ids] = 0
        self.above_steps[env_ids] = 0
        self.episode_number[env_ids] += 1

    def update(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one physics step; return success and timeout masks."""
        self.elapsed_steps += 1
        pose_w = self.object.data.root_link_pose_w.torch
        rotation_w = matrix_from_quat(pose_w[:, 3:7])
        half_size = pose_w.new_tensor(OBJECT_SIZE_M) * 0.5
        bottom_z = pose_w[:, 2] - (rotation_w[:, 2, :].abs() * half_size).sum(dim=-1)
        table_top_z = self.table.data.root_link_pose_w.torch[:, 2] + TABLE_SIZE_M[2] * 0.5
        above = bottom_z >= table_top_z + 0.10
        self.above_steps = torch.where(above, self.above_steps + 1, torch.zeros_like(self.above_steps))
        success = self.above_steps >= self.hold_steps
        timed_out = (self.elapsed_steps >= self.max_steps) & ~success
        return success, timed_out
