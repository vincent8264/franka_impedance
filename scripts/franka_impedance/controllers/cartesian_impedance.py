"""Fixed Cartesian impedance grasp trajectory on the shared effort interface."""

import torch

from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse


class CartesianImpedanceGrasp:
    """Track a nominal grasp with a Jacobian-transpose Cartesian spring and damper.

    The target is planned once from the settled cube pose. Joint torques use the
    same 10%-of-limit action path as the existing scripted physics validation.
    No observed contact force, mass, or friction changes the trajectory or gains.
    """

    settle_steps = 36
    phases = (
        ("approach", 200),
        ("descend", 150),
        ("close", 100),
        ("lift", 200),
        ("hold", 120),
    )

    def __init__(self, env):
        self.env = env
        self.robot = env.robot.data
        self.hand_id = env.observations.ee_body_id
        self.arm_ids = env.control.arm_joint_ids
        self.finger_ids = env.robot.find_bodies(
            ["panda_leftfinger", "panda_rightfinger"], preserve_order=True
        )[0]
        self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.start = env.actions.new_zeros((env.num_envs, 3))
        self.pinch = torch.zeros_like(self.start)
        self.above = torch.zeros_like(self.start)
        self.lifted = torch.zeros_like(self.start)
        self.target = torch.zeros_like(self.start)
        self.down = env.actions.new_tensor([0.0, 1.0, 0.0, 0.0]).expand(env.num_envs, -1)

    def reset(self, env_ids=None):
        """Forget a completed episode before its next settle-and-plan phase."""
        if env_ids is None:
            self.ready[:] = False
            self.target[:] = 0.0
        else:
            self.ready[env_ids] = False
            self.target[env_ids] = 0.0

    @classmethod
    def phase_name(cls, elapsed_step):
        """Name the scheduled phase for a single episode step."""
        if elapsed_step < cls.settle_steps:
            return "settle"
        cursor = cls.settle_steps
        for label, steps in cls.phases:
            cursor += steps
            if elapsed_step < cursor:
                return label
        return "hold"

    def _plan(self, env_ids):
        hand = self.robot.body_link_pose_w.torch[env_ids, self.hand_id]
        fingers = self.robot.body_link_pose_w.torch[env_ids][:, self.finger_ids]
        finger_midpoint = fingers[:, :, :3].mean(dim=1)
        tip_offset = quat_apply_inverse(hand[:, 3:7], finger_midpoint - hand[:, :3])
        # The authored finger pads are 45.25 mm beyond the finger origins.
        tip_offset[:, 2] += 0.04525
        cube_center = self.env.scene["object"].data.root_link_pose_w.torch[env_ids, :3]
        pinch = cube_center - quat_apply(self.down[env_ids], tip_offset)
        self.start[env_ids] = hand[:, :3]
        self.pinch[env_ids] = pinch
        self.above[env_ids] = pinch + hand.new_tensor([0.0, 0.0, 0.12])
        self.lifted[env_ids] = pinch + hand.new_tensor([0.0, 0.0, 0.155])
        self.ready[env_ids] = True

    @staticmethod
    def _smooth_target(start, end, tick, steps):
        fraction = ((tick + 1).float() / (0.8 * steps)).clamp(0.0, 1.0)
        fraction = fraction.square() * (3.0 - 2.0 * fraction)
        return torch.lerp(start, end, fraction[:, None])

    def act(self):
        """Return the eight normalized actions for the current episode steps."""
        elapsed = self.env.episodes.elapsed_steps
        newly_ready = torch.nonzero((elapsed >= self.settle_steps) & ~self.ready).flatten()
        if newly_ready.numel():
            self._plan(newly_ready)

        action = torch.zeros_like(self.env.actions)
        if not self.ready.any():
            return action

        target = self.lifted.clone()
        cursor = self.settle_steps
        for label, steps in self.phases:
            active = self.ready & (elapsed >= cursor) & (elapsed < cursor + steps)
            if label == "approach":
                candidate = self._smooth_target(self.start, self.above, elapsed - cursor, steps)
            elif label == "descend":
                candidate = self._smooth_target(self.above, self.pinch, elapsed - cursor, steps)
            elif label == "close":
                candidate = self.pinch
            else:
                candidate = self._smooth_target(self.pinch, self.lifted, elapsed - cursor, steps) if label == "lift" else self.lifted
            target = torch.where(active[:, None], candidate, target)
            cursor += steps

        self.target[:] = torch.where(self.ready[:, None], target, 0.0)

        hand_pose = self.robot.body_link_pose_w.torch[:, self.hand_id]
        pos_error, rot_error = compute_pose_error(
            hand_pose[:, :3], hand_pose[:, 3:7], target, self.down, rot_error_type="axis_angle"
        )
        jacobian = self.robot.body_link_jacobian_w.torch[:, self.hand_id - 1, :, :][:, :, self.arm_ids]
        joint_velocity = self.robot.joint_vel.torch[:, self.arm_ids]
        twist = (jacobian @ joint_velocity.unsqueeze(-1)).squeeze(-1)
        wrench = torch.cat((300.0 * pos_error, 25.0 * rot_error), dim=-1)
        wrench -= twist * action.new_tensor([30.0, 30.0, 30.0, 5.0, 5.0, 5.0])
        torque = (jacobian.transpose(-1, -2) @ wrench.unsqueeze(-1)).squeeze(-1) - 0.8 * joint_velocity
        scale = self.env.control.limits[:, :7] * self.env.cfg.arm_effort_scale
        action[:, :7] = torch.where(self.ready[:, None], torque / scale, 0.0)
        action[:, 7] = torch.where(self.ready, torch.where(elapsed < self.settle_steps + 350, 1.0, -1.0), 0.0)
        if not torch.isfinite(action).all():
            raise RuntimeError("Cartesian impedance produced a nonfinite action.")
        return action
