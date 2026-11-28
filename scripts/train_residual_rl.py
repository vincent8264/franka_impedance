"""Train bounded residual efforts around the fixed Cartesian impedance grasp."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from franka_impedance.cfg.physics import PHYSICS_SUBSTEPS
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg


def train(env, args):
    import torch
    from franka_impedance.rl.ppo import make_runner

    log_dir = (
        Path(__file__).resolve().parents[1]
        / "logs"
        / "residual_grasp"
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    log_dir.mkdir(parents=True)
    (log_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "num_envs": args.num_envs,
                "iterations": args.iterations,
                "mass_range_kg": args.mass_range,
                "friction_range": args.friction_range,
                "physics_substeps": env.cfg.sim.physics.num_substeps,
                "residual_arm_scale": env.cfg.residual_arm_scale,
                "residual_finger_effort_n": env.cfg.residual_finger_effort_n,
                "residual_cost": env.cfg.residual_cost,
                "num_steps_per_env": 128,
                "gamma": 0.9995,
                "init_std": 0.3,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    trace = []
    runner = make_runner(
        env,
        args.seed,
        args.iterations,
        args.save_interval,
        str(log_dir),
        trace,
        experiment_name="residual_grasp",
        num_steps_per_env=128,
        gamma=0.9995,
        init_std=0.3,
    )
    completed = 0
    if args.resume:
        runner.load(str(args.resume), map_location=env.device)
        completed = runner.current_learning_iteration + 1
        print(f"Resuming after {completed} updates from {args.resume}", flush=True)
    else:
        # Deterministic inference initially reproduces the fixed controller.
        final_layer = runner.alg.get_policy().mlp[-1]
        torch.nn.init.zeros_(final_layer.weight)
        torch.nn.init.zeros_(final_layer.bias)
    runner.save(str(log_dir / "initial.pt"))
    runner.current_learning_iteration = completed
    parameters = list(runner.alg.get_policy().parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    runner.learn(num_learning_iterations=args.iterations)
    assert all(torch.isfinite(parameter).all() for parameter in parameters)
    assert all(
        torch.isfinite(parameter).all() for parameter in runner.alg.critic.parameters()
    )
    assert any(
        not torch.equal(old, new) for old, new in zip(before, parameters, strict=True)
    )
    checkpoint = log_dir / "final.pt"
    runner.save(str(checkpoint))
    curve = (
        torch.stack(trace).reshape(-1, runner.cfg["num_steps_per_env"], 3).mean(dim=1)
    )
    if not torch.isfinite(curve).all():
        raise RuntimeError("Nonfinite residual PPO rollout metrics.")
    with (log_dir / "training.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "iteration",
                "environment_steps",
                "mean_step_reward",
                "arm_residual_saturation_fraction",
                "finger_residual_saturation_fraction",
            )
        )
        writer.writerows(
            (
                i,
                (i - completed) * runner.cfg["num_steps_per_env"] * env.num_envs,
                *metrics,
            )
            for i, metrics in enumerate(curve.tolist(), completed + 1)
        )
    assert checkpoint.stat().st_size > 0
    print(f"Residual PPO checkpoint: {checkpoint}", flush=True)


def main():
    from franka_impedance.envs.object_properties import (
        add_property_args,
        parse_property_ranges,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual-arm-scale", type=float, default=0.02)
    parser.add_argument("--residual-finger-effort", type=float, default=2.0)
    parser.add_argument("--residual-cost", type=float, default=0.002)
    add_property_args(parser, mass_default=(0.5, 1.3), friction_default=(0.08, 0.25))
    add_launcher_args(parser)
    args = parser.parse_args()
    mass_range, friction_range = parse_property_ranges(parser, args)
    if args.num_envs < 1 or args.iterations < 1 or args.save_interval < 1:
        parser.error(
            "Environment count, iterations, and save interval must be positive."
        )
    if args.resume and not args.resume.is_file():
        parser.error(f"Checkpoint does not exist: {args.resume}")

    sim = SimulationCfg(
        dt=1.0 / 120.0,
        device=args.device,
        physics=NewtonCfg(num_substeps=PHYSICS_SUBSTEPS),
    )
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("Residual grasp requires Newton.")
        sim.physics = physics
        from franka_impedance.envs.direct_rl_env import ContactGraspSceneCfg
        from franka_impedance.envs.residual_rl_env import (
            ResidualGraspCfg,
            ResidualGraspEnv,
        )

        cfg = ResidualGraspCfg(
            sim=sim,
            scene=ContactGraspSceneCfg(num_envs=args.num_envs, env_spacing=2.0),
            seed=args.seed,
            object_mass_range=mass_range,
            object_friction_range=friction_range,
            residual_arm_scale=args.residual_arm_scale,
            residual_finger_effort_n=args.residual_finger_effort,
            residual_cost=args.residual_cost,
        )
        env = ResidualGraspEnv(cfg)
        try:
            train(env, args)
        finally:
            env.close()


if __name__ == "__main__":
    main()
