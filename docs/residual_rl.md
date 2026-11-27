# Cartesian impedance with residual RL

The residual policy corrects the fixed [Cartesian impedance grasp](../scripts/franka_impedance/controllers/cartesian_impedance.py). The classical controller still plans the timed approach, descent, closure, lift, and hold trajectory from the settled cube pose. A zero residual follows that controller's effort command. The shared effort interface still applies gravity compensation, finger-limit protection, and actuator clipping.

Training, evaluation, and the fixed baseline now use eight Newton physics substeps per 120 Hz control step. Existing checkpoints and results were generated with one substep; rerun the baseline and train a new policy to compare under this setting.

## RL problem

- **Hidden episode properties:** cube mass and Newton friction. The policy does not receive their sampled values.
- **Observation (58 values):** the shared 46 robot/cube state values, a 3D hand-to-target position error in the robot-base frame, the fixed controller's eight clipped normalized actions, and normalized episode progress. The shared state already includes end-effector and cube poses and velocities.
- **Action (8 values):** seven arm corrections and one driven-finger correction, each clipped to `[-1, 1]`. By default, a unit arm residual adds 2% of that joint's effort limit; a unit finger residual adds 2 N before the finger-limit guard. The fixed arm controller itself uses 10% of the effort limits. Residuals are added before physical actuator clipping, so a saturated fixed action still has correction headroom.
- **Training reward:** the existing contact-aware grasp reward minus `0.002 * dt * mean(residual²)`. Evaluation reports the unmodified task return separately as `mean_episode_return`, alongside the penalized learning return.
- **Termination:** the shared 10 cm table-clearance hold for 0.5 seconds or the 12-second timeout. Contact-verified success remains a separate reported metric.

The actor starts at exactly zero deterministic correction by zero-initializing its last layer. PPO explores with initial standard deviation 0.1, 128 simulation steps per rollout, and discount `gamma=0.9995`. The default training and evaluation distribution is the [challenge range](../results/README.md): 0.5-1.3 kg mass and 0.08-0.25 friction. These are initial settings, not tuned hyperparameters.

## Run from the pinned Isaac Lab checkout

Replace `{project root}` with this repository's absolute path.

```powershell
# Zero residual: check the integration against the fixed controller.
uv run --locked python "{project root}\scripts\eval_residual_rl.py" --episodes 16 --seed 42 --metrics "{project root}\logs\residual_grasp\zero.json" --viz none

# One PPO update checks the training pipeline; use more updates for learning.
uv run --locked python "{project root}\scripts\train_residual_rl.py" --iterations 1 --viz none

# Evaluate the saved policy on the same property distribution.
uv run --locked python "{project root}\scripts\eval_residual_rl.py" --checkpoint "{checkpoint}" --episodes 16 --seed 42 --metrics "{project root}\logs\residual_grasp\evaluation.json" --viz none
```

Training writes `initial.pt`, `final.pt`, periodic checkpoints, a configuration JSON, and a training CSV under `logs/residual_grasp/`. Supply `--resume {checkpoint}` to continue a run. The property flags and residual scales are available on both entry points; keep them matched when evaluating a checkpoint. To compare with direct RL, train it with `--contact-reward --mass-range 0.5 1.3 --friction-range 0.08 0.25` and use held-out evaluation seeds.

## Integration checks

The [zero-residual run](../results/zero_residual_challenge_seed42_16episodes.json) and the fixed [challenge baseline](../results/cartesian_impedance_challenge_seed42_16episodes.json) used the same seed and sampled properties. Both achieved 13/16 contact-verified successes. Mean task return was 0.544281 for zero residual versus 0.540941 for the fixed evaluator; mean episode length was 825.44 versus 824.94 steps. The action composition has a zero-residual equality unit test, but the simulator runs were not stepwise identical.

A one-update PPO smoke run completed 512 environment steps and saved a loadable checkpoint. Its deterministic policy achieved 2/4 on a four-episode evaluation; that is a pipeline check, not a trained performance estimate. Full training and a held-out multi-seed comparison remain to be done.
