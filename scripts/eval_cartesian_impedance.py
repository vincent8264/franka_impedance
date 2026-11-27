"""Evaluate the fixed Cartesian impedance grasp on the shared Direct RL task."""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg

from franka_impedance.cfg.physics import PHYSICS_SUBSTEPS


def evaluate(env, args):
    import torch

    from franka_impedance.controllers.cartesian_impedance import CartesianImpedanceGrasp
    from franka_impedance.envs.object_properties import EpisodePropertyRecord

    controller = CartesianImpedanceGrasp(env)
    env.reset()
    property_record = EpisodePropertyRecord(env)
    gif = None
    if args.gif:
        from franka_impedance.diagnostics.grasp_gif import GraspGif

        gif = GraspGif(env)
        gif.capture(0)
    if args.diagnostics:
        from franka_impedance.diagnostics.grasp_diagnostics import GraspDiagnostics

        env.step_recorder = GraspDiagnostics(env, gif, args.record_steps)

    completed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    quota = (args.episodes + env.num_envs - 1 - torch.arange(env.num_envs, device=env.device)) // env.num_envs
    geometric_successes = torch.zeros_like(completed)
    contact_successes = torch.zeros_like(completed)
    returns = env.actions.new_zeros(env.num_envs)
    return_sum = torch.zeros_like(returns)
    episode_steps = torch.zeros_like(completed)
    step_sum = torch.zeros_like(completed)
    saturated = torch.zeros(8, dtype=torch.long, device=env.device)
    active_steps = torch.zeros((), dtype=torch.long, device=env.device)

    with torch.no_grad():
        for tick in range(env.episodes.max_steps * int(quota.max().item())):
            action = controller.act()
            active = completed < quota
            saturated += ((action.abs() >= 1.0) & active[:, None]).sum(dim=0)
            active_steps += active.sum()
            if args.diagnostics:
                env.step_recorder.raw_action = action
                env.step_recorder.phase = controller.phase_name(int(env.episodes.elapsed_steps[0].item()))
            obs, reward, success, timeout, extras = env.step(action)
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Nonfinite observation or reward during impedance evaluation.")
            done = (success | timeout) & active
            property_record.record(done, completed)
            returns += torch.where(active, reward, 0.0)
            episode_steps += active
            return_sum += torch.where(done, returns, 0.0)
            step_sum += torch.where(done, episode_steps, 0)
            returns.masked_fill_(success | timeout, 0.0)
            episode_steps.masked_fill_(success | timeout, 0)
            geometric_successes += success & done
            contact_successes += extras["grasp_success"] & done
            completed += done
            controller.reset(success | timeout)
            if gif and not args.diagnostics and tick + 1 <= args.record_steps and (tick + 1) % 6 == 0:
                gif.capture(tick + 1)
            if (completed == quota).all():
                break

    total = int(completed.sum().item())
    if total != args.episodes:
        raise RuntimeError(f"Expected {args.episodes} completed episodes, got {total}.")
    metrics = {
        "controller": "cartesian_impedance",
        "seed": args.seed,
        "mass_range_kg": args.mass_range,
        "friction_range": args.friction_range,
        "physics_substeps": env.cfg.sim.physics.num_substeps,
        "episode_properties": sorted(property_record.samples, key=lambda sample: (sample["environment"], sample["episode"])),
        "environments": env.num_envs,
        "episodes": total,
        "geometric_success_rate": int(geometric_successes.sum().item()) / total,
        "contact_verified_success_rate": int(contact_successes.sum().item()) / total,
        "mean_episode_return": float(return_sum.sum().item()) / total,
        "mean_episode_steps": int(step_sum.sum().item()) / total,
        "action_saturation_fraction_by_channel": (saturated / active_steps).tolist(),
        "arm_action_saturation_fraction": float(saturated[:7].sum() / (7 * active_steps)),
        "gripper_action_saturation_fraction": float(saturated[7] / active_steps),
    }
    if args.diagnostics:
        env.step_recorder.save(args.diagnostics)
        env.step_recorder = None
    if gif:
        gif.save(args.gif)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    from franka_impedance.envs.object_properties import add_property_args, parse_property_ranges

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    add_property_args(parser)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--diagnostics", type=Path, help="Write one episode of pre-reset per-step data to CSV.")
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--record-steps", type=int, default=240)
    add_launcher_args(parser)
    args = parser.parse_args()
    mass_range, friction_range = parse_property_ranges(parser, args)
    if min(args.episodes, args.num_envs, args.episode_seconds, args.record_steps) <= 0:
        parser.error("Episode count, environment count, duration, and record steps must be positive.")
    if args.diagnostics and (args.num_envs != 1 or args.episodes != 1):
        parser.error("--diagnostics requires one environment and one episode.")
    if args.gif and args.gif.suffix.lower() != ".gif":
        parser.error("--gif requires a .gif path.")
    sim = SimulationCfg(dt=1.0 / 120.0, device=args.device, physics=NewtonCfg(num_substeps=PHYSICS_SUBSTEPS))
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("Cartesian impedance evaluation requires Newton.")
        sim.physics = physics

        from franka_impedance.envs.direct_rl_env import ContactGraspSceneCfg, DirectGraspCfg, DirectGraspEnv

        cfg = DirectGraspCfg(
            sim=sim,
            scene=ContactGraspSceneCfg(num_envs=args.num_envs, env_spacing=2.0),
            seed=args.seed,
            episode_length_s=args.episode_seconds,
            object_mass_range=mass_range,
            object_friction_range=friction_range,
        )
        env = DirectGraspEnv(cfg)
        try:
            evaluate(env, args)
        finally:
            env.close()


if __name__ == "__main__":
    main()
