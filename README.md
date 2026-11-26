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

Run the project scripts from the Isaac Lab checkout so `uv` uses its environment.
In the commands below, replace `{project root}` with the absolute path to this
repository (for example, `C:\Users\user\Documents\ECE 276C\franka_impedance`).

```powershell
uv run --locked python "{project root}\scripts\run_grasp_scene.py" --viz newton_gl
uv run --locked python "{project root}\scripts\run_grasp_scene.py" --viz none --max_steps 240
```

The runner supports `--gripper-effort`, `--episode-seconds`, and
`--max-episodes`. Episodes default to 12 seconds; success means the cuboid
stays 10 cm above the table for 0.5 seconds. This scene runner uses a posture
hold for inspection.

## Direct RL smoke tests

From the pinned Isaac Lab checkout:

```powershell
uv run --locked python "{project root}\scripts\train_direct_rl.py" --mode rollout --num-envs 2 --viz none
uv run --locked python "{project root}\scripts\train_direct_rl.py" --mode contact --num-envs 1 --viz none
uv run --locked python "{project root}\scripts\train_direct_rl.py" --mode ppo --num-envs 4 --iterations 1 --contact-reward --viz none
```

The policy reads the shared 46-element observation and supplies eight actions
in `[-1, 1]` at 120 Hz. The arm actions scale to 10% of each effort limit
before adding model gravity compensation. The shared effort interface clips
the total command at the joint limits. Gravity compensation offsets the arm's
weight without a position or velocity target. The gripper uses the driven
finger's full effort range with positive effort opening and negative effort
closing the fingers. These scales are initial settings for smoke testing.

The authored cube is 0.2 kg with friction 0.5; episode resets randomize both
properties by default as described below. The rollout test shortens the episode
to one second to exercise timeout
and automatic resets. Its success test moves the cube to a scripted height to
exercise the shared 0.5-second clearance check.

The contact test uses scripted cube and table placements to check Newton's
filtered normal forces. With `--contact-reward`, lift shaping requires
simultaneous cube contact on both fingers, and the completion bonus requires
that contact throughout the 0.5-second clearance window. The shared geometric
success flag is still used for termination. Without the flag, the reward uses
temporary height-only lift and completion terms for smoke testing. PPO logs
and checkpoints go under `logs/direct_grasp/`.

## Object-property variation

The scene, Direct RL trainer, and both evaluators accept `--mass-range MIN MAX`
(kg) and `--friction-range MIN MAX` (Newton's single friction coefficient).
Defaults are `0.1 0.5` kg and `0.2 0.8`, so randomization is enabled. Each
environment samples one mass and one coefficient at every episode reset. A
given seed, environment index, and episode number always produce the same
pair, regardless of when other environments reset. The sampled properties are
not added to the 46-element policy observation. Both evaluators record the
ranges, seed, and each completed episode's sampled values in their JSON metrics.

Train with contact reward and evaluate both baselines over the default
distribution and seed:

```powershell
uv run --locked python "{project root}\\scripts\\train_direct_rl.py" --mode ppo --contact-reward --viz none
uv run --locked python "{project root}\\scripts\\eval_direct_rl.py" --checkpoint "{checkpoint}" --episodes 16 --seed 42 --viz none
uv run --locked python "{project root}\\scripts\\eval_cartesian_impedance.py" --episodes 16 --seed 42 --viz none
```

Newton uses one friction value rather than separate static and dynamic values.
The default ranges are initial robustness settings, not tuned training settings.
Pass `--mass-range 0.2 0.2 --friction-range 0.5 0.5` to reproduce nominal
conditions. The scripted physics validation and gripper diagnostics set these
nominal values explicitly. Use `--mode properties --num-envs 2` to check the
default mass and friction readback from Newton after a partial reset.

## Cartesian impedance baseline

The fixed baseline uses the same Newton scene, task state, contact reward,
effort limits, 120 Hz control rate, and success checks as Direct RL. The scene
still produces the 46-element policy observation; the classical controller
also uses the robot's model Jacobian to turn a Cartesian wrench into torques.
It settles the cube for 36 steps, then approaches, descends, closes, lifts, and
holds on a timed trajectory planned once from the settled cube position. A
Cartesian spring and damper produces arm torques through the hand Jacobian;
model gravity compensation is added by the shared environment. The gripper
requests a fixed +1.05 N opening effort before closure and -17.5 N closing
effort thereafter. The shared finger-limit guard may reduce those efforts near
the joint limits. The gains and timing come from the nominal scripted physics
validation and are not adjusted in response to object mass or friction.

From the pinned Isaac Lab checkout:

```powershell
uv run --locked python "{project root}\scripts\eval_cartesian_impedance.py" --episodes 16 --num-envs 4 --seed 42 --metrics "{project root}\logs\impedance\nominal.json" --viz none
```

For one episode of step-level diagnostics, use `--episodes 1 --num-envs 1
--diagnostics "{project root}\logs\impedance\nominal.csv"`. The evaluation
prints the same success, return, and action-saturation metrics as the Direct RL
evaluator. One nominal seed-42 simulator episode passed both geometric and
contact-verified success checks; see the [development record](docs/direct_rl_development.md).
That single result does not establish robustness across object properties, and
the gripper joint-limit tolerance remains under investigation.

## Entry points

- [run_grasp_scene.py](scripts/run_grasp_scene.py): launch and step the scene.
- [train_direct_rl.py](scripts/train_direct_rl.py): Direct RL rollout, contact probes, and PPO smoke test.
- [eval_direct_rl.py](scripts/eval_direct_rl.py): evaluate a saved Direct RL policy.
- [eval_cartesian_impedance.py](scripts/eval_cartesian_impedance.py): Baseline evaluation and diagnostics.
- [validate_grasp_physics.py](scripts/validate_grasp_physics.py): scripted nominal grasp validation with measured applied effort.
- [diagnose_grasp_handoff.py](scripts/diagnose_grasp_handoff.py) and [diagnose_gripper_limits.py](scripts/diagnose_gripper_limits.py): specialized diagnostics.

Reusable code lives in [scripts/franka_impedance](scripts/franka_impedance):
`cfg` holds the scene, `controllers` the effort and grasp controllers,
`envs` the shared task and Direct RL environment, `rl` the PPO runner,
`diagnostics` the recorders, and `testing` the smoke fixtures. This follows
Isaac Lab's separation of runnable scripts and task modules while keeping the
package importable when scripts are run by path, without an installation step.
