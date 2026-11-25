"""Run one 12-second scripted-to-frozen-policy grasp diagnostic."""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg
from isaaclab_newton.physics import NewtonCfg


def run(env, args):
    import torch
    from tensordict import TensorDict

    from eval_direct_rl import load_actor
    from grasp_diagnostics import GraspDiagnostics
    from grasp_gif import GraspGif
    from validate_grasp_physics import scripted_actions

    actor = load_actor(env, args.checkpoint, 42)
    env.reset()
    gif = GraspGif(env)
    recorder = GraspDiagnostics(env, gif, env.episodes.max_steps)
    env.step_recorder = recorder

    def step(phase, action):
        recorder.phase = phase
        recorder.raw_action = action
        obs, reward, success, timeout, _ = env.step(action)
        if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError("Nonfinite handoff observation or reward.")
        return obs, bool(success[0] | timeout[0])

    with torch.inference_mode():
        zero = torch.zeros_like(env.actions)
        for _ in range(36):
            step("settle", zero)
        for phase, _, action, _ in scripted_actions(env, pregrasp_only=True):
            obs, done = step(f"scripted_{phase}", action)
            if done:
                raise RuntimeError("Episode ended before the policy handoff.")

        # Read the final scripted state. The last 30 descent commands already
        # have a stationary target. No scripted closure or reset follows.
        state = recorder.last_state()
        cube_distance = state["grasp_center_distance_m"]
        left, right = state["left_n"], state["right_n"]
        clearance = state["clearance_m"]
        face_gap = state["signed_pad_face_gap_m"]
        speed = state["hand_speed_mps"]
        angular_speed = state["hand_angular_speed_radps"]
        print(
            f"Handoff after {len(recorder.rows)} steps: center={cube_distance:.5f} m, "
            f"gap={face_gap:.5f} m, speed={speed:.5f} m/s, angular={angular_speed:.5f} rad/s, "
            f"forces=({left:.3f}, {right:.3f}) N, clearance={clearance:.5f} m",
            flush=True,
        )
        if not (
            cube_distance < 0.03 and face_gap > 0.05 and speed < 0.15 and angular_speed < 1.0
            and max(left, right) <= 0.5 and abs(clearance) < 0.005
        ):
            raise RuntimeError("The scripted trajectory did not reach the open, stable pre-grasp pose.")

        for _ in range(env.episodes.max_steps - len(recorder.rows)):
            action = actor(TensorDict(obs, batch_size=[1]), stochastic_output=False)
            obs, done = step("policy_handoff", action)
            if done:
                break
        else:
            raise RuntimeError("Handoff episode did not terminate.")

    recorder.save(args.csv)
    gif.save(args.gif)
    print(f"Saved handoff diagnostic CSV to {args.csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gif", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    sim = SimulationCfg(dt=1.0 / 120.0, device=args.device, physics=NewtonCfg())
    with launch_simulation(sim, args) as physics:
        if not isinstance(physics, NewtonCfg):
            raise RuntimeError("The handoff diagnostic requires Newton.")
        sim.physics = physics
        from direct_rl_env import DirectGraspCfg, DirectGraspEnv
        from smoke_direct_rl import ContactProbeSceneCfg

        cfg = DirectGraspCfg(
            sim=sim, scene=ContactProbeSceneCfg(num_envs=1, env_spacing=2.0), seed=42, episode_length_s=12.0
        )
        env = DirectGraspEnv(cfg)
        try:
            run(env, args)
        finally:
            env.close()


if __name__ == "__main__":
    main()
