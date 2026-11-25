"""Bounded scripted grasp through the unchanged Direct RL action interface."""

import csv
from pathlib import Path

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
    phases = (
        ("approach", 200, pose[:, :3], above, 0.015),
        ("descend", 150, above, pinch, 0.015),
        ("close", 100, pinch, pinch, -0.25),
        ("lift", 200, pinch, lifted, -0.25),
        ("hold", 120, lifted, lifted, -0.25),
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


def validate(env, output: Path):
    """Approach, close, lift and hold a free cube in the nominal scene."""
    env.reset()
    for _ in range(36):
        env.step(torch.zeros_like(env.actions))
    robot = env.robot.data
    hand = env.observations.ee_body_id
    arm = env.control.arm_joint_ids
    limits = env.control.limits
    columns = [
        "phase", "tick", "left_n", "right_n", "left_table_n", "right_table_n",
        "clearance_m", "distance_m", "hand_error_m", "width_m", "reward",
        "reach", "contact", "lift", "action_cost", "completion", "geometric_success",
        "contact_success",
    ]
    for label in ("raw_action", "applied_effort"):
        columns.extend(f"{label}_{joint}" for joint in env.control.joint_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for phase, tick, action, target in scripted_actions(env):
            command = action.clamp(-1., 1.) * limits
            command[:, :7] *= env.cfg.arm_effort_scale
            command[:, :7] += robot.gravity_compensation_forces.torch[:, arm]
            applied = command.clamp(-limits, limits)
            _, reward, done, timeout, extras = env.step(action)
            verified = extras["grasp_success"][0].item()
            # The environment resets on success, so its post-step poses and sensors
            # cannot be used for terminal measurements. The reward is pre-reset.
            if done.item() or timeout.item():
                writer.writerow([phase, tick, *([None] * 8), reward.item(), *([None] * 4),
                                 0.5 * verified, done.item(), verified,
                                 *action[0].tolist(), *applied[0].tolist()])
                print(f"Terminal {phase} tick {tick}: geometric={done.item()}, contact={verified}, timeout={timeout.item()}")
                return verified

            sensors = (*env.cube_sensors, env.scene["left_table"], env.scene["right_table"])
            forces = [env.contact_force(sensor)[0].item() for sensor in sensors]
            fields = env.observations.read()
            distance = (fields["ee_pose_b"][:, :3] - fields["object_pose_b"][:, :3]).norm(dim=-1).item()
            error = (robot.body_link_pose_w.torch[:, hand, :3] - target).norm(dim=-1).item()
            clearance = env.clearance.item()
            grasped = min(forces[:2]) > 0.5
            dt = env.step_dt
            reach = 0.02 * torch.exp(action.new_tensor(-distance / 0.08)).item() * dt
            contact = 0.03 * grasped * dt
            lift = 0.05 * min(1., max(0., clearance / 0.1)) * grasped * dt
            cost = 0.005 * action.clamp(-1., 1.).square().mean().item() * dt
            writer.writerow([phase, tick, *forces, clearance, distance, error,
                             fields["gripper_width"].item(), reward.item(), reach, contact,
                             lift, cost, 0., False, False, *action[0].tolist(), *applied[0].tolist()])
    return False


def main():
    import argparse

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import NewtonCfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    sim = SimulationCfg(dt=1. / 120., device=args.device, physics=NewtonCfg())
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("This validation requires Newton.")
        sim.physics = physics
        from direct_rl_env import DirectGraspCfg, DirectGraspEnv
        from smoke_direct_rl import ContactProbeSceneCfg

        cfg = DirectGraspCfg(sim=sim, scene=ContactProbeSceneCfg(num_envs=1, env_spacing=2.), seed=42)
        env = DirectGraspEnv(cfg)
        try:
            passed = validate(env, args.output)
        finally:
            env.close()
    if not passed:
        raise SystemExit("Scripted contact-verified grasp did not pass.")


if __name__ == "__main__":
    main()
