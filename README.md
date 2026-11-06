# Franka grasp-and-lift project

The [project proposal](docs/proposal.md) describes the goal and the three
controller variants: Cartesian impedance, direct RL, and impedance with an RL
residual.

## Simulator version

Use the Isaac Lab 3.0 development checkout at commit
`e78aa3e21c2d10c0e74b4b28ff537d84f6544fbb`. This exact checkout has been
tested on Windows with the Newton backend and the `Isaac-Lift-Franka` task. The
project code belongs in this repository; do not copy the Isaac Lab source into it.

The commit identifies the Isaac Lab source. The accompanying
[`setup/isaaclab-uv-lock.patch`](setup/isaaclab-uv-lock.patch) records package
version updates needed to make that commit's `uv.lock` match its source package
versions. Keep the commit and patch together when updating the simulator.

## Windows setup for teammates

Install a current NVIDIA driver, Git, and `uv`. Clone this repository and Isaac
Lab as sibling folders, then run the following in PowerShell (replace the team
repository URL):

```powershell
git clone <team-repository-url> franka-grasp-project
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
cd IsaacLab
git checkout e78aa3e21c2d10c0e74b4b28ff537d84f6544fbb
git apply ..\franka-grasp-project\setup\isaaclab-uv-lock.patch
uv sync --locked
uv run --locked isaaclab zero_agent --task Isaac-Lift-Franka --num_envs 1 --max_steps 1 --viz none
```

The final command is a simulator smoke test, not a grasp-success test. Its first
run may spend a few minutes building caches and filling the task's reset bank.
For RSL-RL training, install the optional dependency with
`uv sync --locked --extra rsl-rl`.

Keep task configuration, controllers, training scripts, evaluation scripts, and
results in this repository. Evaluate all three controllers with the same task,
physics backend, object-property distributions, observations, and random seeds.
Generated logs, videos, downloaded datasets, and model checkpoints are ignored
by Git. Small, reviewed result summaries and figures can be committed when
they are ready to share.
