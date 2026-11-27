"""Run a short Direct RL rollout or an RSL-RL PPO smoke test."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg

from franka_impedance.cfg.physics import PHYSICS_SUBSTEPS


def train(env, args):
    import torch

    from franka_impedance.rl.ppo import make_runner

    log_dir = Path(__file__).resolve().parents[1] / "logs" / "direct_grasp" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_dir.mkdir(parents=True)
    (log_dir / "run_config.json").write_text(json.dumps({
        "seed": args.seed, "num_envs": args.num_envs, "iterations": args.iterations,
        "mass_range_kg": args.mass_range, "friction_range": args.friction_range,
        "physics_substeps": env.cfg.sim.physics.num_substeps,
        "contact_reward": args.contact_reward,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"PPO output: {log_dir}")
    trace = []
    runner = make_runner(env, args.seed, args.iterations, args.save_interval, str(log_dir), trace)
    completed = 0
    if args.resume:
        runner.load(str(args.resume), map_location=env.device)
        completed = runner.current_learning_iteration + 1
        print(f"Resuming after {completed} completed updates from {args.resume}", flush=True)
    runner.save(str(log_dir / "initial.pt"))
    # Saved indices denote the update that already ran.
    runner.current_learning_iteration = completed
    parameters = list(runner.alg.get_policy().parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    runner.learn(num_learning_iterations=args.iterations)
    assert all(torch.isfinite(parameter).all() for parameter in parameters)
    assert all(torch.isfinite(parameter).all() for parameter in runner.alg.critic.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters, strict=True))
    checkpoint = log_dir / "final.pt"
    runner.save(str(checkpoint))
    batch_size = runner.cfg["num_steps_per_env"]
    curve = torch.stack(trace).reshape(-1, batch_size, 3).mean(dim=1)
    if not torch.isfinite(curve).all():
        raise RuntimeError("Nonfinite PPO rollout metrics.")
    with (log_dir / "training.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("iteration", "environment_steps", "mean_step_reward",
                         "arm_action_saturation_fraction", "gripper_action_saturation_fraction"))
        writer.writerows((i, (i - completed) * batch_size * env.num_envs, *metrics)
                         for i, metrics in enumerate(curve.tolist(), completed + 1))
    assert checkpoint.stat().st_size > 0
    print(f"PPO checkpoint: {checkpoint}")


def main():
    from franka_impedance.envs.object_properties import (
        NOMINAL_FRICTION, NOMINAL_MASS_KG, add_property_args, parse_property_ranges,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rollout", "contact", "properties", "ppo"), default="rollout")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--resume", type=Path, help="Resume PPO from a completed update checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-reward", action="store_true")
    add_property_args(parser)
    add_launcher_args(parser)
    args = parser.parse_args()
    mass_range, friction_range = parse_property_ranges(parser, args)
    max_iterations = 96 if args.resume else 24
    if not 1 <= args.iterations <= max_iterations or args.save_interval < 1:
        parser.error(f"Use 1 to {max_iterations} iterations and a positive checkpoint interval.")
    if args.resume and (args.mode != "ppo" or not args.resume.is_file()):
        parser.error("--resume requires PPO mode and an existing checkpoint.")
    if args.num_envs < (2 if args.mode == "rollout" else 1):
        parser.error("Use at least two environments for rollout or one for other modes.")
    if args.mode == "properties" and mass_range == (NOMINAL_MASS_KG,) * 2 and friction_range == (NOMINAL_FRICTION,) * 2:
        parser.error("Properties mode requires a nonnominal mass or friction range.")

    sim = SimulationCfg(dt=1.0 / 120.0, device=args.device, physics=NewtonCfg(num_substeps=PHYSICS_SUBSTEPS))
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("Direct grasp requires Newton.")
        sim.physics = physics
        from franka_impedance.envs.direct_rl_env import ContactGraspSceneCfg, DirectGraspCfg, DirectGraspEnv
        from franka_impedance.cfg.grasp_scene_cfg import GraspSceneCfg

        if args.mode == "contact":
            from franka_impedance.testing.smoke_direct_rl import ContactProbeSceneCfg

            scene_type = ContactProbeSceneCfg
        elif args.contact_reward:
            scene_type = ContactGraspSceneCfg
        else:
            scene_type = GraspSceneCfg
        cfg = DirectGraspCfg(
            sim=sim,
            scene=scene_type(num_envs=args.num_envs, env_spacing=2.0),
            seed=args.seed,
            episode_length_s=1.0 if args.mode == "rollout" else 12.0,
            object_mass_range=mass_range,
            object_friction_range=friction_range,
        )
        env = DirectGraspEnv(cfg)
        try:
            if args.mode == "ppo":
                train(env, args)
            else:
                from franka_impedance.testing.smoke_direct_rl import check_contacts, check_properties, check_rollout

                check = {"contact": check_contacts, "properties": check_properties, "rollout": check_rollout}[args.mode]
                check(env)
        finally:
            env.close()


if __name__ == "__main__":
    main()
