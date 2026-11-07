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
stays 10 cm above the table for 0.5 seconds. Grasp control is not implemented
yet.

## Entry points

- [run_grasp_scene.py](scripts/run_grasp_scene.py): launch and step the scene.
- [grasp_scene_cfg.py](scripts/grasp_scene_cfg.py): Franka, table, and cuboid.
- [effort_control.py](scripts/effort_control.py): shared arm torque and gripper effort interface.
- [grasp_task.py](scripts/grasp_task.py): episode resets, success check, and observations.
