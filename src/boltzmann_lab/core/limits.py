"""Shared bounds for the teaching model (UI and calculation engine)."""
import math
from numbers import Integral, Real

from boltzmann_lab.core.config import FIELD_WIDTH, FIELD_HEIGHT, BARRIER_MIN, BARRIER_MAX

MAX_PARTICLES = 2_000
MAX_FILL_FRACTION = 0.25
MAX_CHAMBER_FILL_FRACTION = 0.30
MAX_SCATTER_ROW_FRACTION = 0.25
MAX_POTENTIAL_SAMPLES = 65_536
PLAYBACK_RANGE = (0.05, 8.0)
GAME_COUNT_RANGE = (10, 100)
GAME_TEMPERATURE_RANGE = (0.01, 10.0)
GAME_DURATION_RANGE = (0.1, 120.0)
GAME_HEIGHT_RANGE = (0.1, 12.0)

# Names are the simulation's public parameters, not widget names.
PARAMETER_RANGES = {
    "n": (1, MAX_PARTICLES), "radius_px": (1, 5),
    "temperature": (0.0, 10.0), "energy_scale": (0.0, 12.0),
    "barrier_mode": (0, 5), "barrier_goal": (0.1, 0.9),
    "pore_px": (4, 100), "scatter_shape": (0, 5),
    "scatter_count": (1, 8), "scatter_size_px": (4, 40),
    "scatter_columns": (1, 3), "scatter_angle": (0, 359),
    "p_lr": (0.0, 1.0), "p_rl": (0.0, 1.0),
    "color_count": (1, 3), "demon_color_threshold": (0.0, 40.0),
    "demon_speed_threshold": (0.0, 40.0),
}
INTEGER_PARAMETERS = frozenset({
    "n", "radius_px", "barrier_mode", "pore_px", "scatter_shape",
    "scatter_count", "scatter_size_px", "scatter_columns", "color_count",
})


def require_range(name, value, bounds, integer=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]
            or (integer and not isinstance(value, Integral))):
        raise ValueError(f"{name}: expected {'integer ' if integer else ''}{bounds[0]}..{bounds[1]}")
    return value


def particle_capacity(radius_px, area_fraction=1.0):
    require_range("radius_px", radius_px, PARAMETER_RANGES["radius_px"], integer=True)
    require_range("area_fraction", area_fraction, (0.0, 1.0))
    area = FIELD_WIDTH * FIELD_HEIGHT * area_fraction
    return min(MAX_PARTICLES, max(0, math.floor(MAX_FILL_FRACTION * area / (math.pi * radius_px**2))))


def fill_fraction(count, radius_px, area_fraction=1.0):
    return count * math.pi * radius_px**2 / (FIELD_WIDTH * FIELD_HEIGHT * max(area_fraction, 1e-12))


def radius_capacity(count):
    return max(r for r in range(1, 6) if count <= particle_capacity(r))


def validate_parameters(sim):
    for name, bounds in PARAMETER_RANGES.items():
        require_range(name, getattr(sim, name), bounds, name in INTEGER_PARAMETERS)
    if not math.isclose(sim.radius, sim.radius_px / FIELD_HEIGHT, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("radius: use set_radius_px to change the physical radius")
    if sim.n > particle_capacity(sim.radius_px):
        raise ValueError("particle_density: reduce particle count or radius")
    if (sim.barrier_mode == 3
            and sim.scatter_count * sim.scatter_size_px > FIELD_HEIGHT * MAX_SCATTER_ROW_FRACTION):
        raise ValueError("scatter_geometry: reduce scatterer count or size")
    for name, size, maximum in (("demon_pass", 3, 3), ("demon_else", 3, 1),
                                ("demon_speed_pass", 2, 3), ("demon_speed_else", 2, 1)):
        rules = getattr(sim, name)
        if len(rules) != size:
            raise ValueError(f"{name}: expected {size} rules")
        for rule in rules:
            require_range(name, rule, (0, maximum), integer=True)
    for name in ("demon_rule_mode", "demon_color_mode"):
        choices = ("color", "speed") if name == "demon_rule_mode" else ("random", "speed")
        if getattr(sim, name) not in choices:
            raise ValueError(f"{name}: unsupported mode")


def barrier_motion_limits(sim):
    """Bound compression of sealed gas; open flow directions remain available.

    The initial gas occupies at most 25% of the vessel. The 30% chamber ceiling
    leaves some room for piston experiments while keeping the gas dilute.
    """
    left_closed = right_closed = (sim.barrier_mode == 1 or
                                  (sim.barrier_mode == 2 and sim.pore_px <= 2 * sim.radius_px))
    if sim.barrier_mode == 4:
        left_closed, right_closed = sim.p_lr == 0.0, sim.p_rl == 0.0
    if not (left_closed or right_closed) or len(sim.positions) == 0:
        return BARRIER_MIN, BARRIER_MAX
    left = int((sim.positions[:, 0] < sim.barrier_x).sum())
    disk_area = math.pi * sim.radius_px**2 / (FIELD_WIDTH * FIELD_HEIGHT)
    minimum = max(BARRIER_MIN, left * disk_area / MAX_CHAMBER_FILL_FRACTION) if left_closed else BARRIER_MIN
    maximum = min(BARRIER_MAX, 1.0 - (sim.n - left) * disk_area / MAX_CHAMBER_FILL_FRACTION) if right_closed else BARRIER_MAX
    return min(minimum, BARRIER_MAX), max(maximum, BARRIER_MIN)
