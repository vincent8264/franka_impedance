"""Common eight-channel effort interface for the Franka grasp task.

Channels 0-6 command arm joint torques (N m); channel 7 commands the driven
gripper finger's signed effort (N). The second finger is coupled in the USD
asset and is deliberately not commanded independently.
"""

import torch


ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAME = "panda_finger_joint1"
EFFORT_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]


class FrankaEffortControl:
    """Convert commands to bounded actuator efforts in a fixed joint order."""

    def __init__(self, robot):
        self.robot = robot
        self.joint_ids, joint_names = robot.find_joints(EFFORT_JOINT_NAMES, preserve_order=True)
        if joint_names != EFFORT_JOINT_NAMES:
            raise RuntimeError(f"Expected effort joints {EFFORT_JOINT_NAMES}, got {joint_names}")
        self.arm_joint_ids = self.joint_ids[:7]
        self.limits = robot.data.joint_effort_limits.torch[:, self.joint_ids].clone()
        if not torch.isfinite(self.limits).all() or not (self.limits > 0).all():
            raise RuntimeError(f"Effort limits must be finite and positive: {self.limits}")

    def apply(self, command: torch.Tensor) -> torch.Tensor:
        """Apply physical efforts of shape (num_envs, 8), clipped to joint limits."""
        if command.shape != self.limits.shape:
            raise ValueError(f"Expected effort shape {tuple(self.limits.shape)}, got {tuple(command.shape)}")
        if not torch.isfinite(command).all():
            raise ValueError("Effort command contains NaN or infinity")
        applied = torch.clamp(command, min=-self.limits, max=self.limits)
        self.robot.actuators.target_command.set_effort_index(value=applied, joint_ids=self.joint_ids)
        return applied

    def apply_normalized(self, action: torch.Tensor) -> torch.Tensor:
        """Scale a policy action in [-1, 1]^8 to physical efforts and apply it."""
        if action.shape != self.limits.shape:
            raise ValueError(f"Expected action shape {tuple(self.limits.shape)}, got {tuple(action.shape)}")
        return self.apply(action.clamp(-1.0, 1.0) * self.limits)
