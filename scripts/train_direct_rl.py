"""Run a short Direct RL rollout or an RSL-RL PPO smoke test."""

import argparse
import csv
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg


def make_runner(env, seed, iterations=1, save_interval=1, log_dir=None, step_trace=None):
    import torch

    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
        handle_deprecated_rsl_rl_cfg,
    )
    from rsl_rl.runners import OnPolicyRunner

    class TracedWrapper(RslRlVecEnvWrapper):
        def step(self, actions):
            result = super().step(actions)
            if not (torch.isfinite(result[0]["policy"]).all() & torch.isfinite(result[1]).all()):
                raise RuntimeError("Nonfinite observation or reward during PPO.")
            saturation = (actions.abs() >= 1.0).float()
            step_trace.append(torch.stack((result[1].mean(), saturation[:, :7].mean(), saturation[:, 7].mean())))
            return result

    cfg = RslRlOnPolicyRunnerCfg(
        device=env.device,
        seed=seed,
        num_steps_per_env=24,
        max_iterations=iterations,
        save_interval=save_interval,
        experiment_name="direct_grasp",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlMLPModelCfg(
            hidden_dims=[64, 64],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
        ),
        critic=RslRlMLPModelCfg(hidden_dims=[64, 64], activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=2,
            num_mini_batches=2,
            learning_rate=3e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
    cfg = handle_deprecated_rsl_rl_cfg(cfg, version("rsl-rl-lib"))
    wrapper = RslRlVecEnvWrapper if step_trace is None else TracedWrapper
    return OnPolicyRunner(wrapper(env, clip_actions=1.0), cfg.to_dict(), log_dir, device=env.device)


def train(env, args):
    import torch

    log_dir = Path(__file__).resolve().parents[1] / "logs" / "direct_grasp" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_dir.mkdir(parents=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rollout", "contact", "ppo"), default="rollout")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--resume", type=Path, help="Resume PPO from a completed update checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-reward", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    max_iterations = 96 if args.resume else 24
    if not 1 <= args.iterations <= max_iterations or args.save_interval < 1:
        parser.error(f"Use 1 to {max_iterations} iterations and a positive checkpoint interval.")
    if args.resume and (args.mode != "ppo" or not args.resume.is_file()):
        parser.error("--resume requires PPO mode and an existing checkpoint.")
    if args.num_envs < (2 if args.mode == "rollout" else 1):
        parser.error("Use at least two environments for rollout or one for other modes.")

    sim = SimulationCfg(dt=1.0 / 120.0, device=args.device, physics=NewtonCfg())
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("Direct grasp requires Newton.")
        sim.physics = physics
        from direct_rl_env import ContactGraspSceneCfg, DirectGraspCfg, DirectGraspEnv
        from grasp_scene_cfg import GraspSceneCfg

        if args.mode == "contact":
            from smoke_direct_rl import ContactProbeSceneCfg

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
        )
        env = DirectGraspEnv(cfg)
        try:
            if args.mode == "ppo":
                train(env, args)
            else:
                from smoke_direct_rl import check_contacts, check_rollout

                check = check_contacts if args.mode == "contact" else check_rollout
                check(env)
        finally:
            env.close()


if __name__ == "__main__":
    main()
