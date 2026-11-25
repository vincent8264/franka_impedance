"""RSL-RL PPO runner configuration and saved-actor loading."""

from importlib.metadata import version


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


def load_actor(env, checkpoint, seed):
    import torch

    runner = make_runner(env, seed)
    saved = torch.load(checkpoint, weights_only=True, map_location=env.device)
    actor = runner.get_inference_policy(env.device)
    actor.load_state_dict(saved["actor_state_dict"])
    actor.eval()
    print(f"Loaded deterministic actor from {checkpoint}", flush=True)
    return actor
