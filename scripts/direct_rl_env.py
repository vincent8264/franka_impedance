"""Direct effort control for the shared Franka grasp-and-lift task."""

import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.sensors.contact_sensor import ContactSensorCfg

from effort_control import EFFORT_JOINT_NAMES, FrankaEffortControl
from grasp_rewards import grasp_reward
from grasp_scene_cfg import GraspSceneCfg
from grasp_task import GraspEpisodes, GraspObservations, OBSERVATION_DIM


FINGER_PARENT = (
    "{ENV_REGEX_NS}/Robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/"
    "panda_link4/panda_link5/panda_link6/panda_link7/panda_hand"
)


@configclass
class ContactGraspSceneCfg(GraspSceneCfg):
    left_cube = ContactSensorCfg(
        prim_path=f"{FINGER_PARENT}/panda_leftfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    right_cube = ContactSensorCfg(
        prim_path=f"{FINGER_PARENT}/panda_rightfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


@configclass
class DirectGraspCfg(DirectRLEnvCfg):
    scene = GraspSceneCfg(num_envs=4, env_spacing=2.0)
    sim = SimulationCfg(dt=1.0 / 120.0, physics=NewtonCfg())
    decimation = 1
    episode_length_s = 12.0
    action_space = len(EFFORT_JOINT_NAMES)
    observation_space = OBSERVATION_DIM
    state_space = 0
    arm_effort_scale: float = 0.1
    gravity_compensation: bool = True
    gripper_active_braking: bool = False


class DirectGraspEnv(DirectRLEnv):
    """Isaac Lab advances physics and selects which environments to reset."""

    def __init__(self, cfg: DirectGraspCfg):
        if cfg.decimation != 1:
            raise ValueError("The shared lift tracker requires one physics step per action.")
        if not 0.0 < cfg.arm_effort_scale <= 1.0:
            raise ValueError("Arm effort scale must be in (0, 1].")
        super().__init__(cfg)
        self.robot = self.scene["robot"]
        self.control = FrankaEffortControl(self.robot)
        self.observations = GraspObservations(self.scene)
        self.finger_joint_ids = self.observations.joint_ids[7:9]
        self.episodes = GraspEpisodes(self.scene, self.physics_dt, cfg.episode_length_s)
        self.actions = torch.zeros_like(self.control.limits)
        self.grasp_hold_steps = torch.zeros_like(self.episodes.above_steps)
        self.step_recorder = None
        self.cube_sensors = (
            (self.scene["left_cube"], self.scene["right_cube"])
            if isinstance(cfg.scene, ContactGraspSceneCfg)
            else ()
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != self.actions.shape:
            raise ValueError(f"Expected actions with shape {tuple(self.actions.shape)}, got {tuple(actions.shape)}")
        self.actions.copy_(actions.clamp(-1.0, 1.0))

    def _apply_action(self) -> None:
        command = self.actions * self.control.limits
        command[:, :7] *= self.cfg.arm_effort_scale
        gripper_action = self.actions[:, 7]
        opening = gripper_action > 0.0
        finger_pos = self.robot.data.joint_pos.torch[:, self.finger_joint_ids]
        finger_vel = self.robot.data.joint_vel.torch[:, self.finger_joint_ids]
        finger_limits = self.robot.data.joint_pos_limits.torch[:, self.finger_joint_ids]
        remaining_travel = torch.where(
            opening[:, None], finger_limits[..., 1] - finger_pos, finger_pos - finger_limits[..., 0]
        )
        toward_limit = torch.where(opening[:, None], finger_vel, -finger_vel).clamp_min(0.0)
        # The driven finger moves both coupled fingers. Start reducing effort
        # earlier if either finger is already moving toward the commanded limit.
        projected_travel = remaining_travel - 2.0 * self.physics_dt * toward_limit
        limit_scale = (projected_travel.amin(dim=1) / 0.002).clamp(0.0, 1.0)
        command[:, 7] = gripper_action * torch.where(opening, 1.05, 17.5) * limit_scale
        if self.cfg.gripper_active_braking:
            command[:, 7] = self._brake_gripper(command[:, 7], finger_pos, finger_vel, finger_limits)
        if self.cfg.gravity_compensation:
            # Feedforward model gravity only. The policy supplies all motion commands.
            command[:, :7] += self.robot.data.gravity_compensation_forces.torch[:, self.control.arm_joint_ids]
        applied = self.control.apply(command)
        if self.step_recorder is not None:
            self.step_recorder.capture_effort(command, applied)

    def _brake_gripper(self, effort, position, velocity, limits):
        # Close the velocity envelope over two control steps. Either finger
        # can tighten the effort bound, including when the policy requests zero.
        horizon = 2.0 * self.step_dt
        gain = 24.0  # N s/m, for the coupled fingers at 120 Hz
        max_velocity = (limits[..., 1] - position).clamp_min(0.0) / horizon
        min_velocity = -(position - limits[..., 0]).clamp_min(0.0) / horizon
        lower = (gain * (min_velocity - velocity)).amax(dim=1)
        upper = (gain * (max_velocity - velocity)).amin(dim=1)
        # Only reverse effort to oppose motion at the driven joint. This also
        # resolves competing demands when the fingers move in opposite directions.
        driven_velocity = velocity[:, 0]
        lower = torch.where(driven_velocity < 0.0, lower, lower.clamp_max(0.0))
        upper = torch.where(driven_velocity > 0.0, upper, upper.clamp_min(0.0))
        lower = torch.where(driven_velocity > 0.0, torch.minimum(lower, upper), lower)
        return torch.maximum(torch.minimum(effort, upper), lower).clamp(-17.5, 17.5)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": self.observations.read()["policy"]}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.clearance = self.episodes.clearance()
        return self.episodes.update(self.clearance)

    def _get_rewards(self) -> torch.Tensor:
        obs = self.observations.read()
        distance = torch.linalg.vector_norm(obs["ee_pose_b"][:, :3] - obs["object_pose_b"][:, :3], dim=-1)
        grasped = None
        left = right = None
        completion = self.reset_terminated
        if self.cube_sensors:
            left, right = (self.contact_force(sensor) for sensor in self.cube_sensors)
            grasped = (left > 0.5) & (right > 0.5)
            holding = grasped & (self.episodes.above_steps > 0)
            self.grasp_hold_steps = torch.where(holding, self.grasp_hold_steps + 1, 0)
            completion = completion & (self.grasp_hold_steps >= self.episodes.hold_steps)
            self.extras["grasp_success"] = completion
        reward = grasp_reward(distance, self.clearance, self.actions, completion, self.step_dt, grasped)
        if self.step_recorder is not None:
            self.step_recorder.capture_step(obs, distance, left, right, grasped, completion, reward)
        return reward

    @staticmethod
    def contact_force(sensor) -> torch.Tensor:
        force = sensor.data.normal_force_matrix_w.torch
        return torch.linalg.vector_norm(force, dim=-1).amax(dim=(1, 2))

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        self.episodes.reset(env_ids, reset_scene=False)
        self.actions[env_ids] = 0.0
        self.grasp_hold_steps[env_ids] = 0
