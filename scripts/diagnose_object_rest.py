"""Measure cube motion without robot contact at nominal and randomized mass."""

import argparse
import csv
import json
from pathlib import Path

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg

from franka_impedance.cfg.physics import PHYSICS_SUBSTEPS


def run_case(env, label, mass, friction, inertia, steps, rows, *, baseline=False):
    props = env.episodes.properties
    props.mass_range = (mass, mass)
    props.friction_range = (friction, friction)
    env.reset()
    if inertia is not None:
        env.scene["object"].set_inertias_index(inertias=inertia, env_ids=torch.tensor([0], device=env.device))
    cube = env.scene["object"]
    actual_mass = float(cube.data.body_mass.torch[0, 0])
    actual_inertia = cube.data.body_inertia.torch[0, 0].detach().cpu().tolist()
    samples = []
    zero = torch.zeros_like(env.actions)
    if baseline:
        from franka_impedance.controllers.cartesian_impedance import CartesianImpedanceGrasp

        controller = CartesianImpedanceGrasp(env)
    for tick in range(steps):
        action = controller.act() if baseline else zero
        _, _, success, timeout, _ = env.step(action)
        pose = cube.data.root_link_pose_w.torch[0].detach().cpu()
        velocity = cube.data.root_link_vel_w.torch[0].detach().cpu()
        hand = env.robot.data.body_link_pose_w.torch[0, env.observations.ee_body_id, :3].detach().cpu()
        distance = float(torch.linalg.vector_norm(hand - pose[:3]))
        left, right = [float(env.contact_force(sensor)[0]) for sensor in env.cube_sensors]
        row = [label, tick + 1, *pose.tolist(), *velocity.tolist(), distance, left, right,
               bool(success[0]), bool(timeout[0])]
        rows.append(row)
        samples.append(torch.cat((pose[:3], velocity, torch.tensor([distance, left, right]))))
        if bool(success[0] | timeout[0]):
            raise RuntimeError(f"Unexpected episode reset during {label} at step {tick + 1}")
    samples = torch.stack(samples)
    settled = samples[max(0, steps // 2):]
    speed = torch.linalg.vector_norm(settled[:, 3:6], dim=-1)
    angular_speed = torch.linalg.vector_norm(settled[:, 6:9], dim=-1)
    after_settle = samples[60:]
    no_contact = after_settle[(after_settle[:, 10:12].amax(dim=1) < 0.1) & (after_settle[:, 9] > 0.08)]
    return {
        "case": label,
        "mass_kg": actual_mass,
        "friction": friction,
        "inertia_kg_m2": actual_inertia,
        "minimum_hand_distance_m": float(samples[:, 9].min()),
        "maximum_finger_cube_force_n": float(samples[:, 10:12].max()),
        "last_half_max_linear_speed_mps": float(speed.max()),
        "last_half_rms_linear_speed_mps": float(torch.sqrt(speed.square().mean())),
        "last_half_max_angular_speed_radps": float(angular_speed.max()),
        "last_half_vertical_range_m": float(settled[:, 2].max() - settled[:, 2].min()),
        "last_half_horizontal_range_m": [
            float(settled[:, axis].max() - settled[:, axis].min()) for axis in (0, 1)
        ],
        "after_settle_no_contact_samples": len(no_contact),
        "after_settle_no_contact_max_speed_mps": (
            float(torch.linalg.vector_norm(no_contact[:, 3:6], dim=-1).max()) if len(no_contact) else None
        ),
        "after_settle_no_contact_vertical_range_m": (
            float(no_contact[:, 2].max() - no_contact[:, 2].min()) if len(no_contact) else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=1.1)
    parser.add_argument("--friction", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--mode", choices=("rest", "baseline"), default="rest")
    parser.add_argument("--substeps", type=int, default=PHYSICS_SUBSTEPS)
    parser.add_argument("--collision-decimation", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.mass <= 0 or args.friction <= 0 or args.steps < 2 or args.substeps < 1:
        parser.error("Mass, friction, steps, and substeps must be positive.")
    if args.collision_decimation < 0 or args.collision_decimation >= args.substeps:
        parser.error("Collision decimation must be nonnegative and less than substeps.")
    sim = SimulationCfg(
        dt=1.0 / 120.0, device=args.device,
        physics=NewtonCfg(num_substeps=args.substeps, collision_decimation=args.collision_decimation),
    )
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("Object-rest diagnosis requires Newton.")
        sim.physics = physics
        from franka_impedance.envs.direct_rl_env import ContactGraspSceneCfg, DirectGraspCfg, DirectGraspEnv

        cfg = DirectGraspCfg(
            sim=sim,
            scene=ContactGraspSceneCfg(num_envs=1, env_spacing=2.0),
            seed=42,
            object_mass_range=(0.2, args.mass),
            object_friction_range=(args.friction, 0.5),
        )
        env = DirectGraspEnv(cfg)
        try:
            rows = []
            if args.mode == "rest":
                nominal = run_case(env, "nominal", 0.2, 0.5, None, args.steps, rows)
                nominal_inertia = env.scene["object"].data.body_inertia.torch.clone()
                heavy = run_case(env, "heavy_original_inertia", args.mass, args.friction, None, args.steps, rows)
                corrected = run_case(
                    env, "heavy_scaled_inertia", args.mass, args.friction,
                    nominal_inertia * (args.mass / 0.2), args.steps, rows,
                )
                results = [nominal, heavy, corrected]
            else:
                results = [
                    run_case(env, "nominal_baseline", 0.2, 0.5, None, args.steps, rows, baseline=True),
                    run_case(env, "heavy_baseline", args.mass, args.friction, None, args.steps, rows, baseline=True),
                ]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["case", "step", "x", "y", "z", "qx", "qy", "qz", "qw",
                                 "vx", "vy", "vz", "wx", "wy", "wz", "hand_distance_m",
                                 "left_n", "right_n", "success", "timeout"])
                writer.writerows(rows)
            print(json.dumps(results, indent=2), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
