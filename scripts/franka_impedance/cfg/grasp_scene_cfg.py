"""Minimal tabletop scene for the pinned Isaac Lab 3.0 development checkout."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.core.lift.config.franka.franka_env_cfg import FRANKA_PANDA_LIFT_CFG


TABLE_SIZE_M = (0.8, 1.5, 0.04)
OBJECT_SIZE_M = (0.05, 0.05, 0.05)


TORQUE_FRANKA_CFG = FRANKA_PANDA_LIFT_CFG.copy()
# Effort commands must not be mixed with the asset's built-in position drives.
for actuator_name in ("panda_arm", "panda_hand"):
    TORQUE_FRANKA_CFG.actuators[actuator_name].stiffness = 0.0
    TORQUE_FRANKA_CFG.actuators[actuator_name].damping = 0.0


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """One robot, a collidable tabletop, and a dynamic cuboid per environment."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # The lift task's Franka asset is calibrated for this checkout's Newton backend.
    robot = TORQUE_FRANKA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=TORQUE_FRANKA_CFG.init_state.replace(rot=(0.0, 0.0, 1.0, 0.0)),
    )

    # Top surface is z = 0.255 m. The robot base sits just outside the table edge.
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.235)),
    )

    # Starts 1 cm above the tabletop and settles under gravity.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.1, 0.29)),
    )
