"""Cartesian impedance grasp with bounded learned effort corrections."""

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from franka_impedance.controllers.cartesian_impedance import CartesianImpedanceGrasp
from franka_impedance.controllers.residual_effort import compose_residual_action
from franka_impedance.envs.direct_rl_env import DirectGraspCfg, DirectGraspEnv
from franka_impedance.envs.grasp_task import OBSERVATION_DIM


RESIDUAL_OBSERVATION_DIM = OBSERVATION_DIM + 3 + 8 + 1


@configclass
class ResidualGraspCfg(DirectGraspCfg):
    observation_space = RESIDUAL_OBSERVATION_DIM
    residual_arm_scale: float = 0.02
    residual_finger_effort_n: float = 2.0
    residual_cost: float = 0.002


class ResidualGraspEnv(DirectGraspEnv):
    """Use the fixed trajectory as the zero-residual policy."""

    def __init__(self, cfg: ResidualGraspCfg):
        if not 0.0 < cfg.residual_arm_scale <= 1.0:
            raise ValueError("Residual arm scale must be in (0, 1].")
        if cfg.residual_finger_effort_n <= 0.0 or cfg.residual_cost < 0.0:
            raise ValueError("Residual finger effort must be positive and cost nonnegative.")
        super().__init__(cfg)
        self.controller = CartesianImpedanceGrasp(self)
        self.residual_actions = torch.zeros_like(self.actions)
        self.baseline_actions = torch.zeros_like(self.actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = super()._get_observations()["policy"]
        baseline = self.controller.act().clamp(-1.0, 1.0)
        base_pose = self.robot.data.root_link_pose_w.torch
        hand_pose = self.robot.data.body_link_pose_w.torch[:, self.observations.ee_body_id]
        target_error_b = quat_apply_inverse(
            base_pose[:, 3:7], self.controller.target - hand_pose[:, :3]
        )
        target_error_b = torch.where(self.controller.ready[:, None], target_error_b, 0.0)
        progress = (self.episodes.elapsed_steps.float() / self.episodes.max_steps)[:, None]
        return {"policy": torch.cat((state, target_error_b, baseline, progress), dim=-1)}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != self.residual_actions.shape:
            raise ValueError(f"Expected residual actions with shape {tuple(self.residual_actions.shape)}")
        self.residual_actions.copy_(actions.clamp(-1.0, 1.0))
        self.baseline_actions.copy_(self.controller.act())
        self.actions.copy_(compose_residual_action(
            self.baseline_actions,
            self.residual_actions,
            arm_effort_scale=self.cfg.arm_effort_scale,
            residual_arm_scale=self.cfg.residual_arm_scale,
            residual_finger_effort_n=self.cfg.residual_finger_effort_n,
        ))

    def _get_rewards(self) -> torch.Tensor:
        task_reward = super()._get_rewards()
        self.extras["task_reward"] = task_reward
        penalty = self.step_dt * self.cfg.residual_cost * self.residual_actions.square().mean(dim=-1)
        return task_reward - penalty

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if hasattr(self, "controller"):
            self.controller.reset(env_ids)
            self.residual_actions[env_ids] = 0.0
            self.baseline_actions[env_ids] = 0.0
