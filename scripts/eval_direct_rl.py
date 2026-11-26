"""Evaluate a saved Direct RL actor and optionally record a headless GIF."""

import argparse
import json
from copy import copy
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg


def evaluate(env, args):
    import torch
    from tensordict import TensorDict

    from franka_impedance.rl.ppo import load_actor
    from franka_impedance.envs.object_properties import EpisodePropertyRecord

    actor = load_actor(env, args.checkpoint, args.seed) if args.checkpoint else None
    obs, _ = env.reset()
    property_record = EpisodePropertyRecord(env)
    if args.gif:
        from franka_impedance.diagnostics.grasp_gif import GraspGif

        recorder = GraspGif(env)
    if args.diagnostics:
        from franka_impedance.diagnostics.grasp_diagnostics import GraspDiagnostics

        env.step_recorder = GraspDiagnostics(env, recorder if args.gif else None, args.record_steps)
    completed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    quota = (args.episodes + env.num_envs - 1 - torch.arange(env.num_envs, device=env.device)) // env.num_envs
    cube_successes = torch.zeros_like(completed)
    grasp_successes = torch.zeros_like(completed)
    returns = env.actions.new_zeros(env.num_envs)
    return_sum = torch.zeros_like(returns)
    saturated = torch.zeros(env.actions.shape[1], dtype=torch.long, device=env.device)
    active_steps = torch.zeros((), dtype=torch.long, device=env.device)
    # The environment retains tensors across rollouts and resets them in place.
    with torch.no_grad():
        for tick in range(env.episodes.max_steps * int(quota.max().item())):
            action = (
                actor(TensorDict(obs, batch_size=[env.num_envs]), stochastic_output=False)
                if actor is not None
                else 2 * torch.rand_like(env.actions) - 1
            )
            active = completed < quota
            saturated += ((action.abs() >= 1.0) & active[:, None]).sum(dim=0)
            active_steps += active.sum()
            if args.diagnostics:
                env.step_recorder.raw_action = action
            obs, reward, success, timeout, extras = env.step(action)
            if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Nonfinite observation or reward during evaluation.")
            done = (success | timeout) & (completed < quota)
            property_record.record(done, completed)
            returns += reward
            return_sum += torch.where(done, returns, 0)
            returns.masked_fill_(success | timeout, 0)
            cube_successes += success & done
            grasp_successes += extras["grasp_success"] & done
            completed += done
            if args.gif and not args.diagnostics and tick < args.record_steps and tick % 6 == 0:
                recorder.capture(tick + 1)
            if (completed == quota).all():
                break
    total = int(completed.sum())
    if total != args.episodes:
        raise RuntimeError(f"Expected {args.episodes} completed episodes, got {total}.")
    metrics = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "seed": args.seed,
        "mass_range_kg": args.mass_range,
        "friction_range": args.friction_range,
        "episode_properties": sorted(property_record.samples, key=lambda sample: (sample["environment"], sample["episode"])),
        "environments": env.num_envs,
        "episodes": total,
        "geometric_success_rate": int(cube_successes.sum()) / total,
        "contact_verified_success_rate": int(grasp_successes.sum()) / total,
        "mean_episode_return": float(return_sum.sum()) / total,
        "action_saturation_fraction_by_channel": (saturated / active_steps).tolist(),
        "arm_action_saturation_fraction": float(saturated[:7].sum() / (7 * active_steps)),
        "gripper_action_saturation_fraction": float(saturated[7] / active_steps),
    }
    if args.diagnostics:
        env.step_recorder.save(args.diagnostics)
        env.step_recorder = None
    if args.gif:
        recorder.save(args.gif)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    from franka_impedance.envs.object_properties import add_property_args, parse_property_ranges

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", help="PPO checkpoints in evaluation order.")
    parser.add_argument("--series-output", type=Path, help="Write one JSON per checkpoint and first/last GIFs.")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--episode-seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    add_property_args(parser)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--record-steps", type=int, default=240)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--diagnostics", type=Path, help="Write per-step pre-reset CSV for one checkpoint and episode.")
    add_launcher_args(parser)
    args = parser.parse_args()
    mass_range, friction_range = parse_property_ranges(parser, args)
    if min(args.episodes, args.num_envs, args.episode_seconds, args.record_steps) <= 0:
        parser.error("Episode count, environment count, duration, and record steps must be positive.")
    checkpoints = args.checkpoint or [None]
    for checkpoint in checkpoints:
        if checkpoint is not None and not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
    if len(checkpoints) > 1 and (args.series_output is None or args.gif or args.metrics):
        parser.error("For multiple checkpoints, use --series-output without --gif or --metrics.")
    if args.gif and args.gif.suffix.lower() != ".gif":
        parser.error("--gif requires a .gif path.")
    if args.diagnostics and (
        args.num_envs != 1 or args.episodes != 1 or len(checkpoints) != 1 or checkpoints[0] is None
    ):
        parser.error("--diagnostics requires one environment, one episode, and one checkpoint.")
    sim = SimulationCfg(dt=1.0 / 120.0, device=args.device, physics=NewtonCfg())
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("Direct grasp requires Newton.")
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
            for index, checkpoint in enumerate(checkpoints):
                rollout = copy(args)
                rollout.checkpoint = checkpoint
                if len(checkpoints) > 1:
                    label = f"{index:02d}-{checkpoint.stem}"
                    rollout.metrics = args.series_output / f"{label}.json"
                    if index in (0, len(checkpoints) - 1):
                        rollout.gif = args.series_output / f"{label}-schematic.gif"
                evaluate(env, rollout)
        finally:
            env.close()


if __name__ == "__main__":
    main()
