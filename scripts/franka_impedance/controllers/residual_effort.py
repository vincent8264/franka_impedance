"""Compose bounded residual efforts with the fixed impedance action."""

import torch


OPENING_EFFORT_N = 1.05
CLOSING_EFFORT_N = 17.5


def compose_residual_action(
    baseline: torch.Tensor,
    residual: torch.Tensor,
    *,
    arm_effort_scale: float,
    residual_arm_scale: float,
    residual_finger_effort_n: float,
) -> torch.Tensor:
    """Return the common effort interface's normalized equivalent command.

    Arm corrections are a fraction of each joint's effort limit. The finger
    correction is in newtons and is added before the shared joint-limit guard.
    The result may exceed [-1, 1], allowing a correction at a saturated
    baseline command; the common effort interface still clips physical effort.
    """
    if baseline.shape != residual.shape or baseline.ndim != 2 or baseline.shape[1] != 8:
        raise ValueError("Baseline and residual must have matching (num_envs, 8) shapes.")
    base = baseline.clamp(-1.0, 1.0)
    correction = residual.clamp(-1.0, 1.0)
    arm = base[:, :7] + (residual_arm_scale / arm_effort_scale) * correction[:, :7]
    base_finger = base[:, 7] * torch.where(
        base[:, 7] > 0.0, OPENING_EFFORT_N, CLOSING_EFFORT_N
    )
    finger_effort = base_finger + residual_finger_effort_n * correction[:, 7]
    finger = finger_effort / torch.where(
        finger_effort > 0.0, OPENING_EFFORT_N, CLOSING_EFFORT_N
    )
    return torch.cat((arm, finger[:, None]), dim=-1)
