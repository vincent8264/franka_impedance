"""Launch the Franka tabletop scene.

From the pinned Isaac Lab checkout:
    uv run --locked python ../franka_impedance/scripts/run_grasp_scene.py --viz newton_gl
    uv run --locked python ../franka_impedance/scripts/run_grasp_scene.py --viz none --max_steps 240
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import NewtonCfg

import isaaclab.sim as sim_utils
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka, table, and one graspable cuboid.")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many physics steps; 0 runs until interrupted.")
    parser.add_argument("--max-episodes", type=int, default=0, help="Stop after this many completed episodes; 0 runs indefinitely.")
    parser.add_argument("--episode-seconds", type=float, default=12.0, help="Time limit for each episode (default: 12 s).")
    parser.add_argument("--gripper-effort", type=float, default=0.0, help="Signed driven-finger effort in N (default: 0).")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max_steps must be nonnegative")
    if args.max_episodes < 0 or args.episode_seconds <= 0:
        parser.error("--max-episodes must be nonnegative and --episode-seconds must be positive")
    return args


def run(sim_cfg: sim_utils.SimulationCfg, args: argparse.Namespace) -> None:
    # Runtime imports need to follow launch_simulation, which sets up USD/Kit if needed.
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    from effort_control import FrankaEffortControl
    from grasp_scene_cfg import GraspSceneCfg
    from grasp_task import GraspEpisodes, GraspObservations, OBSERVATION_DIM

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(-1.5, 1.4, 1.2), target=(-0.45, 0.0, 0.25))
    scene = InteractiveScene(GraspSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    control = FrankaEffortControl(robot)
    observations = GraspObservations(scene)
    dt = sim.get_physics_dt()
    episodes = GraspEpisodes(scene, physics_dt=dt, episode_seconds=args.episode_seconds)
    episodes.reset()
    # A low-gain software posture hold keeps the scene viewable without hidden
    # actuator PD. Later controllers will replace this eight-channel command.
    arm_ids = control.arm_joint_ids
    home = robot.data.default_joint_pos.torch[:, arm_ids].clone()
    kp = torch.tensor([40.0, 40.0, 40.0, 40.0, 12.0, 12.0, 8.0], device=sim.device)
    kd = torch.tensor([8.0, 8.0, 8.0, 8.0, 3.0, 3.0, 2.0], device=sim.device)
    command = torch.zeros((scene.num_envs, 8), device=sim.device)
    command[:, 7] = args.gripper_effort
    step = 0
    completed_episodes = 0
    print(f"[INFO] Grasp task ready: {OBSERVATION_DIM} observations, {episodes.max_steps} steps/episode.")
    while args.max_steps == 0 or step < args.max_steps:
        obs = observations.read()
        q = obs["joint_pos"][:, :7]
        qd = obs["joint_vel"][:, :7]
        gravity = robot.data.gravity_compensation_forces.torch[:, arm_ids]
        command[:, :7] = gravity + kp * (home - q) - kd * qd
        control.apply(command)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        step += 1
        success, timed_out = episodes.update()
        done_ids = torch.nonzero(success | timed_out).flatten().tolist()
        for env_id in done_ids:
            result = "success" if bool(success[env_id]) else "timeout"
            print(f"[INFO] Episode {int(episodes.episode_number[env_id])}: {result} after {int(episodes.elapsed_steps[env_id])} steps")
        completed_episodes += len(done_ids)
        if args.max_episodes and completed_episodes >= args.max_episodes:
            break
        episodes.reset(done_ids)


def main() -> None:
    args = parse_args()
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device, physics=NewtonCfg())
    with launch_simulation(sim_cfg, args) as physics_cfg:
        if physics_cfg is None:
            raise RuntimeError("No physics backend was selected.")
        sim_cfg.physics = physics_cfg
        run(sim_cfg, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
