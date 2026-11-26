"""Bounded scripted grasp through the unchanged Direct RL action interface."""

import csv
from pathlib import Path

import torch
from franka_impedance.controllers.scripted_grasp import scripted_actions
from franka_impedance.diagnostics.effort_capture import AppliedEffortCapture


def validate(env, output: Path):
    """Approach, close, lift and hold a free cube in the nominal scene."""
    env.reset()
    for _ in range(36):
        env.step(torch.zeros_like(env.actions))
    robot = env.robot.data
    hand = env.observations.ee_body_id
    columns = [
        "phase", "tick", "left_n", "right_n", "left_table_n", "right_table_n",
        "clearance_m", "distance_m", "hand_error_m", "width_m", "reward",
        "reach", "contact", "lift", "action_cost", "completion", "geometric_success",
        "contact_success",
    ]
    for label in ("raw_action", "applied_effort"):
        columns.extend(f"{label}_{joint}" for joint in env.control.joint_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream, AppliedEffortCapture(env) as effort_capture:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for phase, tick, action, target in scripted_actions(env):
            _, reward, done, timeout, extras = env.step(action)
            applied = effort_capture.applied
            if applied is None:
                raise RuntimeError("The environment did not report applied effort.")
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
        from franka_impedance.envs.direct_rl_env import DirectGraspCfg, DirectGraspEnv
        from franka_impedance.testing.smoke_direct_rl import ContactProbeSceneCfg

        cfg = DirectGraspCfg(
            sim=sim, scene=ContactProbeSceneCfg(num_envs=1, env_spacing=2.), seed=42,
            object_mass_range=(0.2, 0.2), object_friction_range=(0.5, 0.5),
        )
        env = DirectGraspEnv(cfg)
        try:
            passed = validate(env, args.output)
        finally:
            env.close()
    if not passed:
        raise SystemExit("Scripted contact-verified grasp did not pass.")


if __name__ == "__main__":
    main()
