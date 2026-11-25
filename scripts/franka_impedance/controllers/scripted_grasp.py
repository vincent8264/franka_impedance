"""Nominal scripted grasp used for physics validation and diagnostics."""

import torch
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse


def scripted_actions(env, *, pregrasp_only=False):
    """Yield the existing scripted commands from a settled nominal scene."""
    robot = env.robot.data
    hand = env.observations.ee_body_id
    arm = env.control.arm_joint_ids
    fingers = env.robot.find_bodies(["panda_leftfinger", "panda_rightfinger"])[0]
    limits = env.control.limits
    residual_limits = limits[:, :7] * env.cfg.arm_effort_scale
    action = torch.zeros_like(limits)
    pose = robot.body_link_pose_w.torch[:, hand].clone()
    cube = env.scene["object"].data
    center = cube.root_link_pose_w.torch[:, :3].clone()
    finger_midpoint = robot.body_link_pose_w.torch[:, fingers, :3].mean(dim=1)
    tip_offset = quat_apply_inverse(pose[:, 3:7], finger_midpoint - pose[:, :3])
    # The authored finger pads are centered 45.25 mm beyond the finger origins.
    tip_offset[:, 2] += 0.04525
    # Downward hand with closing axis aligned to a face of the reset cube.
    down = pose.new_tensor([[0., 1., 0., 0.]])
    if abs(env.episodes.clearance().item()) > 0.005:
        raise RuntimeError(f"Cube did not settle on the tabletop: clearance={env.episodes.clearance().item():.4f} m")
    pinch = center - quat_apply(down, tip_offset)
    above = pinch + pose.new_tensor([[0., 0., 0.12]])
    lifted = pinch + pose.new_tensor([[0., 0., 0.155]])
    # The current action mapping makes +1 open at 1.05 N and -1 close at 17.5 N.
    phases = (
        ("approach", 200, pose[:, :3], above, 1.0),
        ("descend", 150, above, pinch, 1.0),
        ("close", 100, pinch, pinch, -1.0),
        ("lift", 200, pinch, lifted, -1.0),
        ("hold", 120, lifted, lifted, -1.0),
    )
    for phase, steps, start, end, grip in phases[:2] if pregrasp_only else phases:
        for tick in range(steps):
            ratio = min(1., (tick + 1) / (0.8 * steps))
            target = torch.lerp(start, end, ratio * ratio * (3. - 2. * ratio))
            hand_pose = robot.body_link_pose_w.torch[:, hand]
            pos_err, rot_err = compute_pose_error(
                hand_pose[:, :3], hand_pose[:, 3:7], target, down, rot_error_type="axis_angle"
            )
            jac = robot.body_link_jacobian_w.torch[:, hand - 1, :, :][:, :, arm]
            velocity = robot.joint_vel.torch[:, arm]
            twist = (jac @ velocity.unsqueeze(-1)).squeeze(-1)
            wrench = torch.cat((300. * pos_err, 25. * rot_err), dim=-1)
            wrench -= twist * pose.new_tensor([30., 30., 30., 5., 5., 5.])
            torque = (jac.transpose(-1, -2) @ wrench.unsqueeze(-1)).squeeze(-1) - 0.8 * velocity
            action[:, :7] = torque / residual_limits
            action[:, 7] = grip
            yield phase, tick, action, target
