# Cartesian impedance baseline

Run on 2026-11-26 with project commit `e1a164c` and the pinned Isaac Lab commit
`e78aa3e21c2d10c0e74b4b28ff537d84f6544fbb` (with the repository lockfile
patch). The evaluator used Newton on CUDA, 4 parallel environments, 16 completed
episodes, seed 42, 120 Hz control, one physics substep per control step, and 12-second episode limits. Cube mass was
sampled uniformly from 0.1-0.5 kg and friction from 0.2-0.8. The controller
settings were unchanged.

The current entrypoints use eight substeps. The commands below describe the
historical runs but produce new results under the current setting; keep those
results separate from the JSON files recorded here.

| Metric | Result |
| --- | ---: |
| Geometric success | 16/16 (100%) |
| Contact-verified success | 16/16 (100%) |
| Mean episode return | 0.624500 |
| Mean episode length | 665.75 steps (5.55 s) |
| Mean arm normalized-action saturation | 1.44% |
| Gripper normalized-action saturation | 94.59% |

The full metrics and per-episode sampled properties are in
[cartesian_impedance_seed42_16episodes.json](cartesian_impedance_seed42_16episodes.json).
The gripper saturation metric counts normalized requests at `-1` or `+1`; it
does not measure applied effort after the finger-limit guard.

The shell command returned status 1 after printing the complete metrics, with
no traceback in the saved console log. The JSON was written and validated, but
the shutdown status should be checked if this run is reproduced.

From the Isaac Lab checkout, reproduce the run with:

```powershell
uv run --locked python "{project root}\scripts\eval_cartesian_impedance.py" --episodes 16 --num-envs 4 --seed 42 --metrics "{project root}\results\cartesian_impedance_seed42_16episodes.json" --viz none
```

## Heavier, lower-friction pilot runs

The same controller, seed, 4 environments, 16 episodes, and simulator settings
were used for three additional distributions. Only the mass and friction ranges
changed. Both the geometric and contact-verified success rates matched in each
run. These evaluator runs exited with status 0.

| Distribution | Mass (kg) | Friction | Success | Mean return | Mean steps | Metrics |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Mild | 0.4-1.0 | 0.1-0.3 | 16/16 (100%) | 0.626880 | 676.75 | [JSON](cartesian_impedance_mild_seed42_16episodes.json) |
| Challenge | 0.5-1.3 | 0.08-0.25 | 13/16 (81.25%) | 0.540941 | 824.94 | [JSON](cartesian_impedance_challenge_seed42_16episodes.json) |
| Stress | 0.5-1.5 | 0.05-0.2 | 10/16 (62.5%) | 0.437324 | 965.94 | [JSON](cartesian_impedance_stress_seed42_16episodes.json) |

The challenge range is a candidate for residual RL comparison: the fixed
controller sometimes fails, while most episodes remain solvable by it. The
stress range provides a stronger test, but its hardest combinations may exceed
the available grip force. More episodes and held-out seeds are needed to
estimate success rates reliably and assess any learned improvement.

The [zero-residual integration run](zero_residual_challenge_seed42_16episodes.json)
used the challenge range and the same seed and sampled properties. It also
achieved 13/16 contact-verified successes. Its mean return was 0.544281 and
mean episode length 825.44 steps, close to the fixed evaluator's 0.540941 and
824.94 steps. These runs were not stepwise identical.

To reproduce the challenge run from the Isaac Lab checkout:

```powershell
uv run --locked python "{project root}\scripts\eval_cartesian_impedance.py" --episodes 16 --num-envs 4 --seed 42 --mass-range 0.5 1.3 --friction-range 0.08 0.25 --metrics "{project root}\results\cartesian_impedance_challenge_seed42_16episodes.json" --viz none
```
