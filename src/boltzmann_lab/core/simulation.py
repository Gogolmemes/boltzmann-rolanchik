import numpy as np

from boltzmann_lab.core.config import (
    POTENTIAL_POINTS,
    HIST_BINS,
    FIELD_HEIGHT,
    WORLD_H,
    WORLD_W,
    SUBSTEPS,
    BARRIER_MIN,
    BARRIER_MAX,
    BARRIER_MAX_SPEED,
    BARRIER_RESPONSE,
    MAX_TEMPERATURE,
    TEMPERATURE_CHECK_INTERVAL
)

from boltzmann_lab.core.physics import (
    NUMBA_OK,
    collide_numba,
    Obstacle,
)

class TemperatureLimitExceeded(Exception):
    """Превышена допустимая температура симуляции."""

    def __init__(
        self,
        temperature: float,
        limit: float,
    ):
        self.temperature = temperature
        self.limit = limit

        super().__init__(
            f"Temperature {temperature:.3f} "
            f"exceeded limit {limit:.3f}"
        )

class BoltzmannSimulation:
    """
    Квазидвумерная визуализация частиц.

    Потенциальная энергия зависит только от x:
        U = U(x)

    Поэтому распределение по x является одномерным.

    """

    def __init__(self):
        self.rng = np.random.default_rng()

        self.n = 120

        self.radius_px = 4
        self.radius = self.radius_px / FIELD_HEIGHT

        self.temperature = 0.8
        self.energy_scale = 5.0
        self.gamma = 2.0

        self.barrier_mode = 0
        self.barrier_fraction = 0.5

        self.barrier_goal = 0.5        # куда стенку тянет мышь
        self.barrier_velocity = 0.0    # скорость стенки по x

        self.barrier_does_work = True   # False: стенка «как неподвижная»

        self.color_count = 2
        self.colors = np.zeros(0, dtype=int)
        self.demon_pass = [0, 0, 0]   # 0 не пропускать, 1 →, 2 ←, 3 всегда
        self.demon_else = [0, 0, 0]   # 0 спрашивать, 1 отражать
        self.demon_side = np.zeros(0, dtype=int)   # 0 слева, 1 справа
        self.demon_queue = []           # частицы, ждущие решения

        self.scatter_shape = 0         # 0 круги, 1 эллипсы, 2 треугольники, 3 смешанные
        self.scatter_count = 3
        self.scatter_size_px = 16
        self.scatter_columns = 1
        self.scatter_angle = 0.0       # градусы

        self.p_lr = 0.5                # вероятность пройти слева направо
        self.p_rl = 0.5                # вероятность пройти справа налево

        self._obstacle_key = None
        self._obstacles = []
        
        self.pore_px = 26

        self.potential_defined = False

        self.potential = np.zeros(POTENTIAL_POINTS)
        self.potential_x = np.linspace(
            0.0,
            WORLD_W,
            POTENTIAL_POINTS,
        )

        self.gradient = np.zeros(POTENTIAL_POINTS)

        self.positions = np.zeros((0, 2), dtype=float)
        self.velocities = np.zeros((0, 2), dtype=float)

        self.reset_particles(equilibrium=False)

    # --------------------------------------------------------
    # Потенциал
    # --------------------------------------------------------

    def set_potential(self, potential):
        potential = np.asarray(potential, dtype=float)

        if len(potential) != POTENTIAL_POINTS:
            old_x = np.linspace(0.0, 1.0, len(potential))
            new_x = np.linspace(0.0, 1.0, POTENTIAL_POINTS)

            potential = np.interp(
                new_x,
                old_x,
                potential,
            )

        self.potential = potential.copy()

        self.gradient = np.gradient(
            self.potential,
            self.potential_x,
            edge_order=2,
        )

        # Ограничиваем очень резкие силы после рисования мышью.
        self.gradient = np.clip(
            self.gradient,
            -20.0,
            20.0,
        )

        self.potential_defined = True

    def force_x(self, x):
        grad = np.interp(
            x,
            self.potential_x,
            self.gradient,
        )

        return -self.energy_scale * grad

    # --------------------------------------------------------
    # Частицы
    # --------------------------------------------------------

    def set_radius_px(self, value):
        self.radius_px = int(value)
        self.radius = self.radius_px / FIELD_HEIGHT

        self._apply_outer_walls()

        for _ in range(4):
            self._resolve_collisions()

    def _sample_boltzmann_x(self, count):
        """
        Выбор x с вероятностью exp(-U/T).
        """

        if not self.potential_defined:
            return self.rng.uniform(
                self.radius,
                WORLD_W - self.radius,
                count,
            )

        T = max(self.temperature, 1e-6)

        energy = self.energy_scale * self.potential

        energy = energy - np.min(energy)

        weights = np.exp(
            -np.clip(energy / T, 0.0, 700.0)
        )

        total = np.sum(weights)

        if total <= 0.0 or not np.isfinite(total):
            weights = np.ones_like(weights)

        weights = weights / np.sum(weights)

        indices = self.rng.choice(
            len(self.potential),
            size=count,
            p=weights,
        )

        dx = WORLD_W / (POTENTIAL_POINTS - 1)

        x = self.potential_x[indices]

        x += self.rng.uniform(
            -0.5 * dx,
            0.5 * dx,
            count,
        )

        return np.clip(
            x,
            self.radius,
            WORLD_W - self.radius,
        )

    def reset_particles(self, equilibrium=True):
        """
        Перераспределяет частицы.

        Если потенциал уже существует, x выбирается сразу
        согласно распределению Больцмана.
        """

        if equilibrium and self.potential_defined:
            x = self._sample_boltzmann_x(self.n)
        else:
            x = self.rng.uniform(
                self.radius,
                WORLD_W - self.radius,
                self.n,
            )

        y = self.rng.uniform(
            self.radius,
            WORLD_H - self.radius,
            self.n,
        )

        self.positions = np.column_stack((x, y))

        self.recolor()
        self.demon_assign_sides()

        thermal_speed = np.sqrt(
            max(self.temperature, 1e-6)
        )

        self.velocities = self.rng.normal(
            0.0,
            thermal_speed,
            size=(self.n, 2),
        )

        # Убираем движение центра масс
        self.velocities -= np.mean(
            self.velocities,
            axis=0,
        )

        # Приводим среднюю кинетическую энергию
        # к заданной начальной температуре
        if self.n > 1:
            current_temperature = (
                0.5 * np.mean(
                    np.sum(self.velocities**2, axis=1)
                )
            )

            if current_temperature > 0:
                factor = np.sqrt(
                    self.temperature / current_temperature
                )

                self.velocities *= factor

        self._apply_outer_walls()

        for _ in range(8):
            self._resolve_collisions()

    # --------------------------------------------------------
    # Перегородка
    # --------------------------------------------------------

    @property
    def demon_mode(self):
        return self.barrier_mode == 5

    @property
    def barrier_x(self):
        return self.barrier_fraction * WORLD_W

    @property
    def pore_height(self):
        return self.pore_px / FIELD_HEIGHT

    def pore_intervals(self):
        """
        Поры располагаются по высоте перегородки.
        """

        if self.barrier_mode != 2:
            return []

        pore_h = self.pore_height

        centers = [
            0.20 * WORLD_H,
            0.50 * WORLD_H,
            0.80 * WORLD_H,
        ]

        intervals = []

        for center in centers:
            a = max(0.0, center - pore_h / 2)
            b = min(WORLD_H, center + pore_h / 2)

            intervals.append((a, b))

        # Объединение пересекающихся пор.
        intervals.sort()

        merged = []

        for a, b in intervals:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(
                    merged[-1][1],
                    b,
                )

        return [
            (a, b)
            for a, b in merged
        ]

    def obstacles(self):
        """Список препятствий. Пересобирается только при смене параметров."""
        key = (
            self.scatter_shape,
            self.scatter_count,
            self.scatter_size_px,
            self.scatter_columns,
            self.scatter_angle,
        )

        if key == self._obstacle_key:
            return self._obstacles

        n = self.scatter_count

        # Чтобы соседние препятствия не слипались в сплошную стенку.
        size = min(self.scatter_size_px / FIELD_HEIGHT, 0.25 / n)

        angle = np.radians(self.scatter_angle)

        gradient = self.scatter_shape >= 4
        columns = 5 if gradient else self.scatter_columns
        pitch = (2.4 if gradient else 3.4) * size

        result = []
        k = 0

        for c in range(columns):
            dx = (c - (columns - 1) / 2.0) * pitch

            if c % 2 == 0:
                ys = [(i + 0.5) / n * WORLD_H for i in range(n)]
            else:
                # Сдвинутый ряд, как в доске Гальтона.
                ys = [i / n * WORLD_H for i in range(n + 1)]

            if gradient:
                t = c / (columns - 1)

                if self.scatter_shape == 5:
                    t = 1.0 - t

                # Радиус от 35% до 100% от заданного размера.
                radius = size * (0.35 + 0.65 * t)

            for cy in ys:
                if gradient:
                    ob = Obstacle(
                        "ellipse", dx, cy, angle, a=radius, b=radius
                    )
                else:
                    shape = (
                        self.scatter_shape
                        if self.scatter_shape < 3
                        else k % 3
                    )

                    if shape == 0:
                        ob = Obstacle("ellipse", dx, cy, angle, a=size, b=size)
                    elif shape == 1:
                        ob = Obstacle(
                            "ellipse", dx, cy, angle,
                            a=1.6 * size, b=0.7 * size,
                        )
                    else:
                        ob = Obstacle(
                            "triangle", dx, cy, angle, radius=1.4 * size
                        )

                k += 1
                result.append(ob)

        self._obstacle_key = key
        self._obstacles = result

        return result

    def _collide_scatterers(self):
        """
        Упругое отражение от бесконечно тяжёлых препятствий,
        движущихся вместе с перегородкой со скоростью w вдоль x.
        В системе отсчёта препятствия нормальная компонента скорости
        меняет знак: v' = v - 2 ((v - W) . n) n, где W = (w, 0).
        """
        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        pos = self.positions
        vel = self.velocities

        for ob in self.obstacles():
            c = ob.center(bx)

            near = np.flatnonzero(
                np.sum((pos - c) ** 2, axis=1) < (ob.bound + r) ** 2
            )

            if len(near) == 0:
                continue

            d, normal = ob.distance(pos[near], bx)

            hit = d < r

            if not np.any(hit):
                continue

            idx = near[hit]
            nh = normal[hit]

            pos[idx] += nh * (r - d[hit])[:, None]

            vn = (vel[idx, 0] - w) * nh[:, 0] + vel[idx, 1] * nh[:, 1]

            approaching = vn < 0.0
            j = idx[approaching]

            vel[j] -= 2.0 * vn[approaching][:, None] * nh[approaching]

    def recolor(self):
        self.colors = self.rng.integers(0, self.color_count, self.n)

    def demon_assign_sides(self):
        """Запоминает, по какую сторону линии находится каждая частица."""
        self.demon_side = (self.positions[:, 0] >= self.barrier_x).astype(int)
        self.demon_queue.clear()

    def _demon_place(self, i, side):
        eps = 1e-6
        self.positions[i, 0] = self.barrier_x + (eps if side == 1 else -eps)

    def _demon_reflect(self, i):
        side = int(self.demon_side[i])
        self._demon_place(i, side)

        speed = abs(self.velocities[i, 0])
        self.velocities[i, 0] = speed if side == 1 else -speed

    def _demon_collision(self):
        """
        Удар = центр частицы пересёк линию. Правило по цвету:
        пропустить, отразить или поставить частицу в очередь на решение.
        Ждущая частица возвращается на свою сторону, скорость не меняется.
        """
        bx = self.barrier_x
        x = self.positions[:, 0]
        side = self.demon_side

        crossed = (
            ((side == 0) & (x >= bx))
            | ((side == 1) & (x < bx))
        )

        for i in np.flatnonzero(crossed):
            c = int(self.colors[i])
            mode = self.demon_pass[c]

            # side == 0: частица летит слева направо (→), side == 1: справа налево (←)
            allowed = (
                mode == 3
                or (mode == 1 and side[i] == 0)
                or (mode == 2 and side[i] == 1)
            )

            if allowed:
                side[i] = 1 - side[i]
            elif self.demon_else[c] == 1:
                self._demon_reflect(i)
            else:
                self._demon_place(i, side[i])
                self.demon_queue.append(int(i))

    def _demon_enforce(self):
        """После парных столкновений никто не должен проскочить без решения."""
        if len(self.demon_side) != len(self.positions):
            return

        bx = self.barrier_x
        x = self.positions[:, 0]

        left = (self.demon_side == 0) & (x >= bx)
        right = (self.demon_side == 1) & (x < bx)

        self.positions[left, 0] = bx - 1e-6
        self.positions[right, 0] = bx + 1e-6

    def demon_resolve(self, let_pass):
        """Решение пользователя для первой частицы в очереди."""
        if not self.demon_queue:
            return

        i = self.demon_queue.pop(0)

        if let_pass:
            side = 1 - int(self.demon_side[i])
            self.demon_side[i] = side
            self._demon_place(i, side)
        else:
            self._demon_reflect(i)

    def _semipermeable_crossing(self, old_x, old_bx):
        """
        Каждое пересечение плоскости перегородки — случайное решение.
        Прошла: скорость и положение не меняются.
        Не прошла: отражение от стенки, v' = 2w - v.
        """
        bx = self.barrier_x
        w = self.collision_wall_velocity()
        
        x = self.positions[:, 0]

        left_to_right = (old_x < old_bx) & (x >= bx)
        right_to_left = (old_x >= old_bx) & (x < bx)

        hit = left_to_right | right_to_left

        if not np.any(hit):
            return

        probability = np.where(left_to_right, self.p_lr, self.p_rl)
        rejected = hit & (self.rng.random(self.n) >= probability)

        for mask, side in (
            (rejected & left_to_right, -1.0),
            (rejected & right_to_left, +1.0),
        ):
            self.positions[mask, 0] = bx + side * 1e-6

            v = self.velocities[:, 0]

            # Отражаем только если частица и стенка сближаются.
            if side < 0.0:
                bounce = mask & (v > w)
            else:
                bounce = mask & (v < w)

            self.velocities[bounce, 0] = 2.0 * w - v[bounce]

    def can_pass_array(self, y):
        """Векторная версия particle_can_pass."""
        y = np.asarray(y)

        if self.barrier_mode == 0:
            return np.ones(len(y), dtype=bool)

        if self.barrier_mode == 1:
            return np.zeros(len(y), dtype=bool)

        result = np.zeros(len(y), dtype=bool)
        r = self.radius

        if self.pore_height <= 2.0 * r:
            return result

        for a, b in self.pore_intervals():
            result |= (y - r >= a) & (y + r <= b)

        return result

    def particle_can_pass(self, y):
        if self.barrier_mode == 0:
            return True

        if self.barrier_mode == 1:
            return False

        # Диаметр должен помещаться в пору.
        if self.pore_height <= 2.0 * self.radius:
            return False

        for a, b in self.pore_intervals():
            if (
                y - self.radius >= a
                and y + self.radius <= b
            ):
                return True

        return False

    def collision_wall_velocity(self):
        """Скорость стенки, которая входит в формулы столкновения."""
        return self.barrier_velocity if self.barrier_does_work else 0.0

    def _move_barrier(self, dt):
        """Плавно двигает стенку к цели и запоминает её скорость."""
        goal = float(np.clip(self.barrier_goal, BARRIER_MIN, BARRIER_MAX))

        move = (goal - self.barrier_fraction) * WORLD_W
        move *= min(1.0, dt * BARRIER_RESPONSE)

        limit = BARRIER_MAX_SPEED * dt
        move = float(np.clip(move, -limit, limit))

        self.barrier_fraction += move / WORLD_W
        self.barrier_velocity = move / dt

    def _barrier_collision(self, old_x, old_bx):
        """
        Столкновение с бесконечно тяжёлой подвижной стенкой.

        Сторону определяем по положению ДО шага, чтобы быстрая стенка
        не могла «проскочить» сквозь частицу.

        В системе отсчёта стенки нормальная компонента скорости
        меняет знак: v' - w = -(v - w)  =>  v' = 2w - v.
        """
        if self.barrier_mode == 0:
            return

        if self.barrier_mode == 3:
            self._collide_scatterers()
            return

        if self.barrier_mode == 4:
            self._semipermeable_crossing(old_x, old_bx)
            return

        if self.barrier_mode == 5:
            self._demon_collision()
            return

        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        blocked = ~self.can_pass_array(self.positions[:, 1])

        x = self.positions[:, 0]
        v = self.velocities[:, 0]

        left = blocked & (old_x < old_bx)
        right = blocked & (old_x >= old_bx)

        hit = left & (x > bx - r)
        self.positions[hit, 0] = bx - r
        bounce = hit & (v > w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

        hit = right & (x < bx + r)
        self.positions[hit, 0] = bx + r
        bounce = hit & (v < w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

    def _enforce_barrier(self):
        """Страховка после столкновений частиц друг с другом."""

        if self.barrier_mode == 5:
            self._demon_enforce()
            return

        if self.barrier_mode in (0, 3, 4):
            return

        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        blocked = ~self.can_pass_array(self.positions[:, 1])

        x = self.positions[:, 0]
        v = self.velocities[:, 0]

        near = blocked & (np.abs(x - bx) < r)

        left = near & (x < bx)
        self.positions[left, 0] = bx - r
        bounce = left & (v > w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

        right = near & (x >= bx)
        self.positions[right, 0] = bx + r
        bounce = right & (v < w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

    # --------------------------------------------------------
    # Стенки
    # --------------------------------------------------------

    def _apply_outer_walls(self):
        if len(self.positions) == 0:
            return

        r = self.radius

        # Левая стенка
        mask = self.positions[:, 0] < r
        self.positions[mask, 0] = r
        self.velocities[mask, 0] = np.abs(
            self.velocities[mask, 0]
        )

        # Правая стенка
        mask = self.positions[:, 0] > WORLD_W - r
        self.positions[mask, 0] = WORLD_W - r
        self.velocities[mask, 0] = -np.abs(
            self.velocities[mask, 0]
        )

        # Верх
        mask = self.positions[:, 1] < r
        self.positions[mask, 1] = r
        self.velocities[mask, 1] = np.abs(
            self.velocities[mask, 1]
        )

        # Низ
        mask = self.positions[:, 1] > WORLD_H - r
        self.positions[mask, 1] = WORLD_H - r
        self.velocities[mask, 1] = -np.abs(
            self.velocities[mask, 1]
        )

    # --------------------------------------------------------
    # Столкновения частиц
    # --------------------------------------------------------

    def _resolve_collisions(self):
        if self.n <= 1 or self.radius <= 0:
            return

        pos = self.positions
        vel = self.velocities

        use_numba = (
            NUMBA_OK
            and pos.dtype == np.float64
            and vel.dtype == np.float64
            and pos.flags.c_contiguous
            and vel.flags.c_contiguous
        )

        if not use_numba:
            self._resolve_collisions_python()
            return

        collide_numba(pos, vel, self.radius, WORLD_W, WORLD_H)

        self._apply_outer_walls()
        self._enforce_barrier()

    def _resolve_collisions_python(self):
        """
        Упругие столкновения одинаковых круглых частиц.

        Для ускорения используется пространственная сетка,
        а не полный перебор всех N^2 пар.
        """

        if self.n <= 1:
            return

        diameter = 2.0 * self.radius

        if diameter <= 0:
            return

        cell_size = max(
            diameter * 1.05,
            1e-6,
        )

        grid = {}

        for i, position in enumerate(self.positions):
            cell = (
                int(position[0] / cell_size),
                int(position[1] / cell_size),
            )

            grid.setdefault(cell, []).append(i)

        checked = set()

        neighbour_offsets = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),  (0, 0),  (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ]

        min_distance_sq = diameter * diameter

        for cell, particles in grid.items():
            cx, cy = cell

            candidates = []

            for ox, oy in neighbour_offsets:
                candidates.extend(
                    grid.get(
                        (cx + ox, cy + oy),
                        [],
                    )
                )

            for i in particles:
                for j in candidates:
                    if j <= i:
                        continue

                    pair = (i, j)

                    if pair in checked:
                        continue

                    checked.add(pair)

                    delta = (
                        self.positions[j]
                        - self.positions[i]
                    )

                    distance_sq = float(
                        np.dot(delta, delta)
                    )

                    if (
                        distance_sq >= min_distance_sq
                        or distance_sq <= 1e-16
                    ):
                        continue

                    distance = np.sqrt(distance_sq)
                    normal = delta / distance

                    overlap = diameter - distance

                    # Раздвигаем частицы.
                    self.positions[i] -= (
                        normal * overlap * 0.5
                    )

                    self.positions[j] += (
                        normal * overlap * 0.5
                    )

                    # Относительная скорость вдоль нормали.
                    relative = float(
                        np.dot(
                            self.velocities[j]
                            - self.velocities[i],
                            normal,
                        )
                    )

                    # Они движутся навстречу друг другу.
                    if relative < 0.0:
                        self.velocities[i] += (
                            relative * normal
                        )

                        self.velocities[j] -= (
                            relative * normal
                        )

        self._apply_outer_walls()
        self._enforce_barrier()

    # --------------------------------------------------------
    # Ланжевеновская динамика
    # --------------------------------------------------------

    def step(self, dt):
        if not self.potential_defined:
            return

        old_bx = self.barrier_x
        if self.demon_mode:
            # Линия неподвижна и ставится строго по координате.
            self.barrier_fraction = float(
                np.clip(self.barrier_goal, BARRIER_MIN, BARRIER_MAX)
            )
            self.barrier_velocity = 0.0
        else:
            self._move_barrier(dt)

        force = self.force_x(self.positions[:, 0])
        self.velocities[:, 0] += 0.5 * dt * force

        old_x = self.positions[:, 0].copy()
        self.positions += self.velocities * dt

        self._apply_outer_walls()
        self._barrier_collision(old_x, old_bx)

        force_new = self.force_x(self.positions[:, 0])
        self.velocities[:, 0] += 0.5 * dt * force_new

        self.check_temperature_limit()

        self._resolve_collisions()

    # --------------------------------------------------------
    # Статистика
    # --------------------------------------------------------

    def histogram(self, bins=HIST_BINS):
        if self.n == 0:
            return (
                np.zeros(bins),
                np.linspace(0, 1, bins + 1),
            )

        normalized_x = (
            self.positions[:, 0] / WORLD_W
        )

        counts, edges = np.histogram(
            normalized_x,
            bins=bins,
            range=(0.0, 1.0),
        )

        probabilities = (
            counts.astype(float) / self.n
        )

        return probabilities, edges

    def theoretical_distribution(self, bins=HIST_BINS):
        centers = (
            np.arange(bins) + 0.5
        ) / bins

        potential = np.interp(
            centers,
            np.linspace(
                0.0,
                1.0,
                POTENTIAL_POINTS,
            ),
            self.potential,
        )

        T = max(self.temperature, 1e-8)

        energy = self.energy_scale * potential
        energy -= np.min(energy)

        weights = np.exp(
            -np.clip(
                energy / T,
                0.0,
                700.0,
            )
        )

        # Если перегородка сплошная, число частиц
        # с каждой стороны сохраняется.
        # Поэтому нормируем теорию отдельно по двум областям.
        if self.barrier_mode in (1, 4):
            barrier = self.barrier_fraction

            left_mask = centers < barrier
            right_mask = ~left_mask

            current_left = float(
                np.mean(self.positions[:, 0] < self.barrier_x)
            )

            if self.barrier_mode == 4:
                # Стационарное состояние: потоки через стенку равны,
                # N_L p_LR / Z_L = N_R p_RL / Z_R.
                zl = float(np.sum(weights[left_mask]))
                zr = float(np.sum(weights[right_mask]))

                a = self.p_rl * zl
                b = self.p_lr * zr

                if a + b > 0.0:
                    current_left = a / (a + b)

            current_right = 1.0 - current_left

            result = np.zeros_like(weights)

            if np.any(left_mask):
                left_weights = weights[left_mask]

                if np.sum(left_weights) > 0:
                    result[left_mask] = (
                        current_left
                        * left_weights
                        / np.sum(left_weights)
                    )

            if np.any(right_mask):
                right_weights = weights[right_mask]

                if np.sum(right_weights) > 0:
                    result[right_mask] = (
                        current_right
                        * right_weights
                        / np.sum(right_weights)
                    )

            return centers, result

        total = np.sum(weights)

        if total <= 0:
            weights[:] = 1.0
            total = np.sum(weights)

        return centers, weights / total

    def center_of_mass(self):
        if self.n == 0:
            return 0.0

        return float(
            np.mean(self.positions[:, 0])
            / WORLD_W
        )

    def entropy(self):
        """
        Дискретная крупнозернистая энтропия:
            S = -sum p_i ln(p_i)

        k_B = 1.
        """

        p, _ = self.histogram()

        p = p[p > 0.0]

        if len(p) == 0:
            return 0.0

        return float(
            -np.sum(p * np.log(p))
        )

    def effective_temperature(self):
        if self.n == 0:
            return 0.0

        v2 = np.sum(self.velocities**2, axis=1)

        return float(0.5 * np.mean(v2))

    def check_temperature_limit(self):
        temperature = self.effective_temperature()

        if not np.isfinite(temperature):
            raise TemperatureLimitExceeded(
                temperature,
                MAX_TEMPERATURE,
            )

        if temperature > MAX_TEMPERATURE:
            raise TemperatureLimitExceeded(
                temperature,
                MAX_TEMPERATURE,
            )