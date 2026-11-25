# Franka grasp-and-lift project

The [project proposal](docs/proposal.md) describes the goal and the three
controller variants: Cartesian impedance, direct RL, and impedance with an RL
residual.

## Setup (Windows)

Use Isaac Lab commit `e78aa3e21c2d10c0e74b4b28ff537d84f6544fbb` with the
[lockfile patch](setup/isaaclab-uv-lock.patch). Install an NVIDIA driver, Git,
and `uv`. For sibling checkouts:

```powershell
git clone https://github.com/vincent8264/franka_impedance.git franka_impedance
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
cd IsaacLab
git checkout e78aa3e21c2d10c0e74b4b28ff537d84f6544fbb
git apply ..\franka_impedance\setup\isaaclab-uv-lock.patch
uv sync --locked
uv run --locked isaaclab zero_agent --task Isaac-Lift-Franka --num_envs 1 --max_steps 1 --viz none
```

For RL training dependencies: `uv sync --locked --extra rsl-rl`.

## Run the project scene

Run from the Isaac Lab checkout so `uv` uses its environment. If the checkouts
are not siblings, replace the script path with its absolute path.

```powershell
uv run --locked python ..\franka_impedance\scripts\run_grasp_scene.py --viz newton_gl
uv run --locked python ..\franka_impedance\scripts\run_grasp_scene.py --viz none --max_steps 240
```

The runner supports `--gripper-effort`, `--episode-seconds`, and
`--max-episodes`. Episodes default to 12 seconds; success means the cuboid
stays 10 cm above the table for 0.5 seconds. This scene runner uses a posture
hold for inspection.

## Direct RL smoke tests

From the pinned Isaac Lab checkout, use the existing virtual environment:

```powershell
& ./.venv/Scripts/python.exe -B -u ../franka_impedance/scripts/train_direct_rl.py --mode rollout --num-envs 2 --viz none
& ./.venv/Scripts/python.exe -B -u ../franka_impedance/scripts/train_direct_rl.py --mode contact --num-envs 1 --viz none
& ./.venv/Scripts/python.exe -B -u ../franka_impedance/scripts/train_direct_rl.py --mode ppo --num-envs 4 --iterations 1 --contact-reward --viz none
```

The policy reads the shared 46-element observation and supplies eight actions
in `[-1, 1]` at 120 Hz. The arm actions scale to 10% of each effort limit
before adding model gravity compensation. The shared effort interface clips
the total command at the joint limits. Gravity compensation offsets the arm's
weight without a position or velocity target. The gripper uses the driven
finger's full effort range with positive effort opening and negative effort
closing the fingers. These scales are initial settings for smoke testing.

The nominal scene uses the shared 0.2 kg cube with static and dynamic friction
of 0.5. The rollout test shortens the episode to one second to exercise timeout
and automatic resets. Its success test moves the cube to a scripted height to
exercise the shared 0.5-second clearance check.

The contact test uses scripted cube and table placements to check Newton's
filtered normal forces. With `--contact-reward`, lift shaping requires
simultaneous cube contact on both fingers, and the completion bonus requires
that contact throughout the 0.5-second clearance window. The shared geometric
success flag is still used for termination. Without the flag, the reward uses
temporary height-only lift and completion terms for smoke testing. PPO logs
and checkpoints go under `logs/direct_grasp/`.

## Entry points

- [run_grasp_scene.py](scripts/run_grasp_scene.py): launch and step the scene.
- [grasp_scene_cfg.py](scripts/grasp_scene_cfg.py): Franka, table, and cuboid.
- [effort_control.py](scripts/effort_control.py): shared arm torque and gripper effort interface.
- [grasp_task.py](scripts/grasp_task.py): episode resets, success check, and observations.
- [train_direct_rl.py](scripts/train_direct_rl.py): Direct RL rollout, contact probes, and PPO smoke test.
