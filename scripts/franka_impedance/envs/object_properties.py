"""Repeatable per-episode cube mass and Newton friction sampling."""

import math

import torch


NOMINAL_MASS_KG = 0.2
NOMINAL_FRICTION = 0.5
DEFAULT_MASS_RANGE_KG = (0.1, 0.5)
DEFAULT_FRICTION_RANGE = (0.2, 0.8)


def add_property_args(parser, mass_default=DEFAULT_MASS_RANGE_KG,
                      friction_default=DEFAULT_FRICTION_RANGE) -> None:
    parser.add_argument("--mass-range", type=float, nargs=2, metavar=("MIN_KG", "MAX_KG"),
                        default=mass_default,
                        help=f"Uniform cube mass range in kg; default {mass_default[0]} {mass_default[1]}")
    parser.add_argument("--friction-range", type=float, nargs=2, metavar=("MIN_MU", "MAX_MU"),
                        default=friction_default,
                        help=f"Uniform Newton friction coefficient range; default {friction_default[0]} {friction_default[1]}")


def parse_property_ranges(parser, args) -> tuple[tuple[float, float], tuple[float, float]]:
    mass_range = tuple(args.mass_range)
    friction_range = tuple(args.friction_range)
    try:
        validate_range("mass", mass_range)
        validate_range("friction", friction_range)
    except ValueError as error:
        parser.error(str(error))
    return mass_range, friction_range


def validate_range(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not 0.0 < bounds[0] <= bounds[1]:
        raise ValueError(f"{name} range must contain finite, positive, ordered values: {bounds}")


def _uniform(seed: int, env_ids: torch.Tensor, episode_numbers: torch.Tensor, salt: int) -> torch.Tensor:
    """A stateless 31-bit mixer; each (seed, environment, episode) has a fixed draw."""
    value = (env_ids.to(torch.int64) * 73856093 + episode_numbers.to(torch.int64) * 19349663
             + seed * 83492791 + salt) & 0x7FFFFFFF
    value = ((value ^ (value >> 16)) * 1103515245) & 0x7FFFFFFF
    value = ((value ^ (value >> 15)) * 1103515245) & 0x7FFFFFFF
    return value.to(torch.float64) / 2147483648.0


def sample_properties(seed: int, env_ids: torch.Tensor, episode_numbers: torch.Tensor,
                      mass_range: tuple[float, float], friction_range: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one mass and one friction value per cube, independent of reset order."""
    mass = mass_range[0] + (mass_range[1] - mass_range[0]) * _uniform(seed, env_ids, episode_numbers, 0x13579BDF)
    friction = friction_range[0] + (friction_range[1] - friction_range[0]) * _uniform(
        seed, env_ids, episode_numbers, 0x2468ACE0
    )
    return mass.to(torch.float32), friction.to(torch.float32)


class NewtonObjectProperties:
    """Write sampled properties through the pinned Isaac Lab/Newton runtime views."""

    def __init__(self, cube, num_envs: int, device: str, seed: int,
                 mass_range: tuple[float, float], friction_range: tuple[float, float]):
        from isaaclab_newton.physics import NewtonManager
        from newton import ModelFlags
        import warp as wp

        validate_range("mass", mass_range)
        validate_range("friction", friction_range)
        self.cube = cube
        self.seed = seed
        self.mass_range = mass_range
        self.friction_range = friction_range
        self.mass_kg = torch.full((num_envs,), NOMINAL_MASS_KG, device=device)
        self.friction = torch.full((num_envs,), NOMINAL_FRICTION, device=device)
        self._manager = NewtonManager
        self._shape_flag = ModelFlags.SHAPE_PROPERTIES
        self._wp = wp
        self._friction_binding = cube._root_view.get_attribute("shape_material_mu", NewtonManager.get_model())[:, 0]

    def reset(self, env_ids: torch.Tensor, episode_numbers: torch.Tensor) -> None:
        mass, friction = sample_properties(
            self.seed, env_ids, episode_numbers, self.mass_range, self.friction_range
        )
        mass = mass.to(self.mass_kg.device)
        friction = friction.to(self.friction.device)
        self.cube.set_masses_index(masses=mass[:, None], env_ids=env_ids)
        material = self._wp.to_torch(self._friction_binding)
        material[env_ids] = friction[:, None]
        self._manager.add_model_change(self._shape_flag)
        self.mass_kg[env_ids] = mass
        self.friction[env_ids] = friction


class EpisodePropertyRecord:
    """Retain pre-reset samples while the environment automatically starts new episodes."""

    def __init__(self, env):
        self.properties = env.episodes.properties
        self.mass_kg = (self.properties.mass_kg.clone() if self.properties is not None
                        else torch.full((env.num_envs,), NOMINAL_MASS_KG, device=env.device))
        self.friction = (self.properties.friction.clone() if self.properties is not None
                         else torch.full((env.num_envs,), NOMINAL_FRICTION, device=env.device))
        self.samples = []

    def record(self, done: torch.Tensor, completed: torch.Tensor) -> None:
        ids = torch.nonzero(done).flatten()
        for env_id in ids.tolist():
            self.samples.append({
                "environment": env_id,
                "episode": int(completed[env_id]),
                "mass_kg": float(self.mass_kg[env_id]),
                "friction": float(self.friction[env_id]),
            })
        if self.properties is not None and ids.numel():
            self.mass_kg[ids] = self.properties.mass_kg[ids]
            self.friction[ids] = self.properties.friction[ids]
