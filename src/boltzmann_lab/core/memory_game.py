"""A blind experiment: infer which well initially contained the gas."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from boltzmann_lab.core.config import POTENTIAL_POINTS, WORLD_H, WORLD_W
from boltzmann_lab.core.simulation import BoltzmannSimulation

@dataclass(frozen=True)
class Well:
    minimum: float
    left: float
    right: float

@dataclass(frozen=True)
class RoundResult:
    temperature: float
    duration: float
    well_count: int
    origin: int
    guess: int
    retained: int
    particle_count: int
    @property
    def correct(self):
        return self.origin == self.guess


def preset_potential(wells):
    if wells not in (2, 3, 4):
        raise ValueError("bad_preset")
    return np.cos(wells * np.pi * np.linspace(0.0, 1.0, POTENTIAL_POINTS)) ** 2


def find_wells(potential):
    """Find basins separated by substantial maxima, ignoring drawing jitter."""
    values = np.asarray(potential, dtype=float)
    if values.ndim != 1 or len(values) < 32 or not np.isfinite(values).all():
        raise ValueError("need_wells")
    span = float(np.ptp(values))
    if span < 1e-8:
        raise ValueError("need_wells")
    size = min(21, max(3, len(values) // 50))
    size += 1 - size % 2
    smooth = np.convolve(np.pad(values, size // 2, mode="edge"), np.ones(size) / size, mode="valid")
    delta = np.diff(smooth)
    indices = np.flatnonzero(np.abs(delta) > span * 1e-10)
    if len(indices) < 2:
        raise ValueError("need_wells")
    minima = [0] if delta[indices[0]] > 0 else []
    for a, b in zip(indices, indices[1:]):
        if delta[a] < 0 < delta[b]:
            minima.append((int(a) + 1 + int(b)) // 2)
    if delta[indices[-1]] < 0:
        minima.append(len(values) - 1)
    changed = True
    while changed and len(minima) > 1:
        changed = False
        for i, (a, b) in enumerate(zip(minima, minima[1:])):
            prominence = float(np.max(smooth[a:b + 1]) - max(smooth[a], smooth[b]))
            if prominence < 0.04 * span or (b - a) / (len(values) - 1) < 0.07:
                del minima[i if smooth[a] > smooth[b] else i + 1]
                changed = True
                break
    if len(minima) < 2:
        raise ValueError("need_wells")
    if len(minima) > 6:
        raise ValueError("too_many_wells")
    boundaries = [0.0]
    for a, b in zip(minima, minima[1:]):
        boundaries.append((a + int(np.argmax(smooth[a:b + 1]))) / (len(values) - 1))
    boundaries.append(1.0)
    return [Well(m / (len(values) - 1), left, right)
            for m, left, right in zip(minima, boundaries, boundaries[1:])]


class MemoryGame:
    def __init__(self, seed=None):
        self.sim = BoltzmannSimulation()
        self.sim.rng = np.random.default_rng(seed)
        self.phase = "setup"
        self.wells = []
        self.origin = None
        self.initial_positions = None
        self.elapsed = 0.0
        self.duration = 5.0
        self.result = None
        self.results = deque(maxlen=200)
        self.played = self.correct = self.streak = 0
        self.chance_sum = 0.0

    def set_potential(self, potential):
        if self.phase in ("evolving", "guess"):
            raise ValueError("round_active")
        wells = find_wells(potential)
        values = np.asarray(potential, dtype=float)
        values = (values - values.min()) / np.ptp(values)
        self.sim.set_potential(values)
        self.wells = wells
        self.clear_round()

    def clear_round(self):
        self.phase = "setup"
        self.origin = None
        self.initial_positions = None
        self.elapsed = 0.0
        self.result = None

    def well_at(self, fraction):
        for i, well in enumerate(self.wells):
            if well.left <= fraction <= well.right:
                return i
        raise ValueError("bad_choice")

    def _pack_positions(self, well, count, temperature):
        radius = self.sim.radius
        left, right = well.left * WORLD_W + radius, well.right * WORLD_W - radius
        if right - left < 2 * radius:
            raise ValueError("narrow_well")
        grid = np.linspace(left, right, 4096)
        energy = np.interp(grid, self.sim.potential_x, self.sim.potential) * self.sim.energy_scale
        weights = np.exp(np.clip(-(energy - energy.min()) / temperature, -700, 0))
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        positions = np.empty((count, 2), dtype=float)
        placed = 0
        for _ in range(30000):
            point = np.array([np.interp(self.sim.rng.random(), cdf, grid),
                              self.sim.rng.uniform(radius, WORLD_H - radius)])
            if placed and np.any(np.sum((positions[:placed] - point) ** 2, axis=1) < (2.02 * radius) ** 2):
                continue
            positions[placed] = point
            placed += 1
            if placed == count:
                return positions
        raise ValueError("packing_failed")

    def start(self, temperature=0.3, duration=5.0, count=60, height=1.0, origin=None):
        if self.phase not in ("setup", "result"):
            raise ValueError("round_active")
        if not self.wells:
            raise ValueError("need_wells")
        if (not np.isfinite([temperature, duration, height]).all()
                or not 0.01 <= temperature <= 10 or not 0.1 <= duration <= 120
                or not 0.1 <= height <= 12 or not 2 <= count <= 200):
            raise ValueError("bad_parameters")
        if origin is None:
            origin = int(self.sim.rng.integers(len(self.wells)))
        if not isinstance(origin, (int, np.integer)) or not 0 <= origin < len(self.wells):
            raise ValueError("bad_choice")
        self.sim.barrier_mode = 0
        self.sim.set_radius_px(2)
        self.sim.n = int(count)
        self.sim.temperature = float(temperature)
        self.sim.energy_scale = float(height)
        self.sim.reset_particles(equilibrium=False)
        self.sim.positions = self._pack_positions(self.wells[origin], count, temperature)
        self.sim.demon_assign_sides()
        self.initial_positions = self.sim.positions.copy()
        self.origin = int(origin)
        self.duration = float(duration)
        self.elapsed = 0.0
        self.result = None
        self.phase = "evolving"

    def advance(self, dt):
        if self.phase != "evolving":
            return
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("bad_parameters")
        step = min(dt, self.duration - self.elapsed)
        if step > 0:
            self.sim.step(step)
            self.elapsed += step
        if self.duration - self.elapsed < 1e-10:
            self.elapsed = self.duration
            self.phase = "guess"

    def answer(self, guess):
        if self.phase != "guess":
            raise ValueError("round_active")
        if not isinstance(guess, (int, np.integer)) or not 0 <= guess < len(self.wells):
            raise ValueError("bad_choice")
        well = self.wells[self.origin]
        x = self.sim.positions[:, 0] / WORLD_W
        retained = int(np.count_nonzero((x >= well.left) & (x <= well.right)))
        result = RoundResult(self.sim.temperature, self.duration, len(self.wells),
                             self.origin, int(guess), retained, self.sim.n)
        self.result = result
        self.results.append(result)
        self.played += 1
        self.correct += int(result.correct)
        self.streak = self.streak + 1 if result.correct else 0
        self.chance_sum += 1 / len(self.wells)
        self.phase = "result"
        return result
