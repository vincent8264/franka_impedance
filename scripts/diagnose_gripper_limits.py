"""Compare bounded gripper efforts from the scripted pre-grasp pose."""

import argparse
import csv
import json
from itertools import islice
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg


def run(env, output):
    import torch

    from grasp_diagnostics import GraspDiagnostics
    from smoke_direct_rl import move_asset
    from validate_grasp_physics import scripted_actions

    finger_joints = env.observations.joint_ids[7:9]
    limits = env.robot.data.joint_pos_limits.torch[0, finger_joints].tolist()
    effort_limit = env.control.limits[0, 7].item()
    cases = (
        ("script_open", 0.015, True),
        ("full_open", 1.0, True),
        ("script_close", -0.25, True),
        ("full_close", -1.0, True),
        ("full_open_cube_away", 1.0, False),
        ("full_close_cube_away", -1.0, False),
    )
    rows = []
    with torch.no_grad():
        for case, grip, cube_present in cases:
            env.step_recorder = None
            env.reset()
            zero = torch.zeros_like(env.actions)
            for _ in range(36):
                env.step(zero)
            trajectory = scripted_actions(env)
            for phase, _, action, _ in islice(trajectory, 350):
                env.step(action)

            recorder = GraspDiagnostics(env, None, 0)
            pregrasp = env.observations.read()
            if case == cases[0][0]:
                reference = pregrasp["policy"].clone()
            else:
                torch.testing.assert_close(pregrasp["policy"], reference, rtol=0.0, atol=1e-6)
            if not cube_present:
                cube = env.scene["object"]
                pose = cube.data.root_link_pose_w.torch.clone()
                pose[:, 1] += 0.5
                move_asset(cube, torch.arange(env.num_envs, device=env.device), pose)
            env.step_recorder = recorder
            recorder.phase = case
            for tick, (phase, _, action, _) in enumerate(islice(trajectory, 60), 1):
                assert phase == "close"
                action[:, 7] = grip
                recorder.raw_action = action
                _, _, success, timeout, _ = env.step(action)
                if (success | timeout).any():
                    raise RuntimeError("Unexpected terminal step during effort pulse.")
                state = recorder.last_state()
                left_q, right_q = state["left_joint_m"], state["right_joint_m"]
                violation = lambda q, bound: max(bound[0] - q, q - bound[1], 0.0)
                rows.append({
                    "case": case, "tick": tick, "time_s": tick * env.step_dt,
                    "effort_n": recorder.applied[0, 7].item(),
                    "left_joint_m": left_q, "right_joint_m": right_q,
                    "left_limit_violation_m": violation(left_q, limits[0]),
                    "right_limit_violation_m": violation(right_q, limits[1]),
                    **{key: state[key] for key in (
                        "width_m", "pad_center_separation_m", "signed_pad_face_gap_m",
                        "left_n", "right_n", "grasp_center_distance_m", "clearance_m",
                    )},
                    "left_table_n": env.contact_force(env.scene["left_table"]).item(),
                    "right_table_n": env.contact_force(env.scene["right_table"]).item(),
                })
            print(f"{case}: captured {tick} steps", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    with (output / "effort_pulses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "seed": 42, "dt_s": env.step_dt, "pregrasp_steps": 386, "pulse_steps": 60,
        "finger_position_limits_m": dict(zip(("left", "right"), limits, strict=True)),
        "left_command_limit_n": effort_limit,
        "both_joint_effort_limits_n": env.robot.data.joint_effort_limits.torch[0, finger_joints].tolist(),
        "stiffness": env.robot.data.joint_stiffness.torch[0, finger_joints].tolist(),
        "damping": env.robot.data.joint_damping.torch[0, finger_joints].tolist(),
        "native_newton_actuators": env.cfg.sim.use_newton_actuators,
        "pregrasp_joint_positions_m": reference[0, 7:9].tolist(),
    }
    (output / "effort_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    sim = SimulationCfg(dt=1 / 120, device=args.device, physics=NewtonCfg())
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("The gripper diagnostic requires Newton.")
        sim.physics = physics
        from direct_rl_env import DirectGraspCfg, DirectGraspEnv
        from smoke_direct_rl import ContactProbeSceneCfg

        env = DirectGraspEnv(DirectGraspCfg(
            sim=sim, scene=ContactProbeSceneCfg(num_envs=1, env_spacing=2.0), seed=42,
        ))
        try:
            run(env, args.output)
        finally:
            env.close()


if __name__ == "__main__":
    main()
