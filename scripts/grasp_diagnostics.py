"""Capture single-episode diagnostics after physics and before automatic reset."""

import csv

import torch
from isaaclab.utils.math import quat_apply

from effort_control import EFFORT_JOINT_NAMES
from grasp_rewards import grasp_reward_terms


class GraspDiagnostics:
    state_columns = (
        "tick", "time_s", "phase", "distance_m", "grasp_center_distance_m", "left_n", "right_n",
        "clearance_m", "width_m", "reward", "reach", "contact", "lift", "action_cost",
        "completion", "above_steps", "grasp_hold_steps", "geometric_success", "contact_success",
        "timeout", "pad_center_separation_m", "signed_pad_face_gap_m",
        "hand_speed_mps", "hand_angular_speed_radps", "left_joint_m", "right_joint_m",
    )

    def __init__(self, env, gif, record_steps):
        self.env = env
        self.gif = gif
        self.record_steps = record_steps
        self.rows = []
        self.phase = "policy"
        self.fingers = env.robot.find_bodies(
            ["panda_leftfinger", "panda_rightfinger"], preserve_order=True
        )[0]
        # Authored pad center and inward face in each finger's local frame.
        self.pad_offset = env.actions.new_tensor([[0.0, 0.00758, 0.04525]]).expand(2, -1)
        self.face_offset = env.actions.new_tensor([[0.0, 0.00758 - 0.00760, 0.04525]]).expand(2, -1)
        self.open_axis = env.actions.new_tensor([[0.0, 1.0, 0.0]])

    def capture_effort(self, command, applied):
        self.command = command
        self.applied = applied

    def capture_step(self, obs, distance, left, right, grasped, completion, reward):
        env = self.env
        tick = len(self.rows) + 1
        dt = env.step_dt
        hand = env.robot.data.body_link_pose_w.torch[:, env.observations.ee_body_id]
        fingers = env.robot.data.body_link_pose_w.torch[0, self.fingers]
        pads = fingers[:, :3] + quat_apply(fingers[:, 3:7], self.pad_offset)
        faces = fingers[:, :3] + quat_apply(fingers[:, 3:7], self.face_offset)
        center = pads.mean(dim=0)
        pad_separation = (pads[0] - pads[1]).norm().reshape(1)
        axis = quat_apply(hand[:, 3:7], self.open_axis)
        face_gap = ((faces[0] - faces[1]) * axis).sum(dim=-1)
        cube = env.scene["object"].data.root_link_pose_w.torch[:, :3]
        center_distance = (center - cube).norm(dim=-1)
        reach, contact, lift, cost = grasp_reward_terms(distance, env.clearance, env.actions, grasped)
        scalars = torch.stack(
            (distance, center_distance, left, right, env.clearance, obs["gripper_width"][:, 0],
             reward, reach * dt, contact * dt, lift * dt, cost * dt, 0.5 * completion,
             env.episodes.above_steps, env.grasp_hold_steps,
             env.reset_terminated, completion, env.reset_time_outs,
             pad_separation, face_gap,
             obs["ee_vel_b"][:, :3].norm(dim=-1), obs["ee_vel_b"][:, 3:].norm(dim=-1)),
            dim=-1,
        )
        values = torch.cat(
            (scalars, obs["joint_pos"][:, 7:9], self.raw_action, env.actions, self.command, self.applied), dim=-1
        )
        self.rows.append((tick, tick * dt, self.phase, *values[0].tolist()))
        if self.gif and tick <= self.record_steps:
            terminal = env.reset_terminated[0] | env.reset_time_outs[0]
            if tick % 6 == 0 or tick == self.record_steps or terminal:
                self.gif.capture(tick)

    def last_state(self):
        return dict(zip(self.state_columns, self.rows[-1][:len(self.state_columns)], strict=True))

    def save(self, path):
        columns = list(self.state_columns)
        for label in ("raw_action", "normalized_action", "command_effort", "applied_effort"):
            columns.extend(f"{label}_{joint}" for joint in EFFORT_JOINT_NAMES)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(self.rows)
