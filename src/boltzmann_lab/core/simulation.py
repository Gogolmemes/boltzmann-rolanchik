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
    DT,
)

from boltzmann_lab.core.physics import (
    NUMBA_OK,
    collide_numba,
    Obstacle, energy_totals, kick_drift, finish_kick, outer_walls, stabilize_energy, scatter_collisions,
)

from boltzmann_lab.core.limits import (
    PARAMETER_RANGES, MAX_POTENTIAL_SAMPLES, particle_capacity, require_range,
    validate_parameters, barrier_motion_limits,
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
        self._scatter_kernel_key = None
        self._collision_workspace = None
        self._old_x = np.empty(0)
        self._validated_configuration = None
        self._potential_revision = 0
        self._theory_cache = {}
        self.rng = np.random.default_rng()  # Генератор случайных чисел (для разброса частиц)

        self.n = 120  # Начальное количество частиц в симуляции

        self.radius_px = 4  # Радиус частицы на экране (в пикселях)
        self.radius = self.radius_px / FIELD_HEIGHT  # Радиус частицы в физических единицах

        self.temperature = 0.8  # Температура системы (влияет на скорость движения)
        self.energy_scale = 5.0  # Множитель для потенциальной энергии (чтобы ямы/горы были выше/ниже)
        self.gamma = 2.0  # Коэффициент трения (сейчас в коде почти не используется)

        self.barrier_mode = 0  # Тип перегородки (0 - нет, 1 - сплошная, 2 - с порами, 3 - препятствия)
        self.barrier_fraction = 0.5  # Начальное положение перегородки (0.5 = ровно посередине)

        self.barrier_goal = 0.5  # куда стенку тянет мышь
        self.barrier_velocity = 0.0  # скорость стенки по x

        self.barrier_does_work = True  # False: стенка «как неподвижная»

        self.demon_rule_mode = "color"
        self.demon_speed_threshold = 1.0
        # Индексы скоростных правил: 0 — медленные (v <= порога), 1 — быстрые.
        self.demon_speed_pass = [2, 1]
        self.demon_speed_else = [1, 1]
        self.demon_color_mode = "random"
        self.demon_color_threshold = 1.0
        self.color_count = 2  # Сколько цветов у частиц (для Демона Максвелла)
        self.colors = np.zeros(0, dtype=int)  # Массив, где хранится цвет каждой частицы
        self.demon_pass = [0, 0, 0]  # 0 не пропускать, 1 →, 2 ←, 3 всегда
        self.demon_else = [0, 0, 0]  # 0 спрашивать, 1 отражать
        self.demon_side = np.zeros(0, dtype=int)  # 0 слева, 1 справа
        self.demon_queue = []  # частицы, ждущие решения

        self.scatter_shape = 0  # 0 круги, 1 эллипсы, 2 треугольники, 3 смешанные
        self.scatter_count = 3  # Сколько препятствий создать по высоте
        self.scatter_size_px = 16  # Размер препятствий (в пикселях)
        self.scatter_columns = 1  # Сколько колонок с препятствиями
        self.scatter_angle = 0.0  # градусы

        self.p_lr = 0.5  # вероятность пройти слева направо
        self.p_rl = 0.5  # вероятность пройти справа налево

        self._obstacle_key = None  # Служебный ключ (чтобы не пересоздавать препятствия каждый кадр)
        self._obstacles = []  # Список созданных объектов препятствий

        self.pore_px = 26  # Размер отверстий (пор) в перегородке (в пикселях)

        self.potential_defined = False  # Нарисован ли уже график потенциальной энергии (False - нет)

        self.potential = np.zeros(POTENTIAL_POINTS)  # Массив значений энергии U(x) (пока нули)
        self.potential_x = np.linspace(0.0, WORLD_W, POTENTIAL_POINTS)  # Координаты X для каждой точки энергии

        self.gradient = np.zeros(POTENTIAL_POINTS)  # Производная энергии (сила), посчитанная для каждой точки

        self.positions = np.zeros((0, 2), dtype=float)  # Координаты X и Y всех частиц (пока пусто)
        self.velocities = np.zeros((0, 2), dtype=float)  # Скорости Vx и Vy всех частиц (пока пусто)

        self.reset_particles(equilibrium=False)  # Запуск: создаем частицы и расставляем их случайным образом

    # --------------------------------------------------------
    # Потенциал
    # --------------------------------------------------------

    def set_potential(self, potential): # вызывается каждый раз, когда вы рисуете график потенциальной энергии U(x) мышкой
        # Его задача — взять нарисованную кривую, превратить её в массив чисел, вычислить силы и сохранить всё это в память программы.
        potential = np.asarray(potential, dtype=float)
        if (potential.ndim != 1 or not 2 <= potential.size <= MAX_POTENTIAL_SAMPLES
                or not np.isfinite(potential).all()
                or np.min(potential) < -1e-12 or np.max(potential) > 1.0 + 1e-12):
            raise ValueError("potential: expected a finite one-dimensional curve in 0..1")

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
        self._potential_revision += 1
        self._theory_cache.clear()

    def force_x(self, x): # сила — это минус производная потенциальной энергии
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
        require_range("radius_px", value, PARAMETER_RANGES["radius_px"], integer=True)
        if self.n > particle_capacity(value):
            raise ValueError("particle_density: reduce particle count before increasing radius")
        self.radius_px = int(value)
        self.radius = self.radius_px / FIELD_HEIGHT

        self._apply_outer_walls()

        for _ in range(4):
            self._resolve_collisions()

    def _sample_boltzmann_x(self, count): # задача — расставить частицы в пространстве не случайно, а так, как требует физика, то есть с вероятностью e^−U(x)/T
 
        # Если вы нажмете кнопку «Распределить заново», программа вызовет именно эту функцию
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
        ) # Вычисление вероятностей

        total = np.sum(weights) # Нормировка 

        if total <= 0.0 or not np.isfinite(total):
            weights = np.ones_like(weights)

        weights = weights / np.sum(weights)

        indices = self.rng.choice(
            len(self.potential),
            size=count,
            p=weights,
        ) # Случайный выбор точек

        dx = WORLD_W / (POTENTIAL_POINTS - 1)

        x = self.potential_x[indices]

        x += self.rng.uniform(
            -0.5 * dx,
            0.5 * dx,
            count,
        ) # Добавление микрослучайности

        return np.clip(
            x,
            self.radius,
            WORLD_W - self.radius,
        ) # Защита от стенок

    def reset_particles(self, equilibrium=True): # вызывается при старте программы, при нажатии кнопки «Распределить заново» или при изменении количества частиц.
        # задача — создать все частицы заново, расставить их по экрану и задать им правильные скорости
        """
        Перераспределяет частицы.

        Если потенциал уже существует, x выбирается сразу
        согласно распределению Больцмана.
        """

        validate_parameters(self)
        if equilibrium and self.potential_defined:
            x = self._sample_boltzmann_x(self.n)
        else:
            x = self.rng.uniform( # Расстановка по горизонтали
                self.radius,
                WORLD_W - self.radius,
                self.n,
            )

        y = self.rng.uniform(
            self.radius,
            WORLD_H - self.radius,
            self.n,
        )

        self.positions = np.column_stack((x, y)) # Сохранение координат

        self.recolor()
        self.demon_assign_sides() # запоминает, кто из частиц находится слева от перегородки, а кто справа. Это тоже нужно для Демона

        thermal_speed = np.sqrt(
            max(self.temperature, 1e-6)
        )

        self.velocities = self.rng.normal( # Центр масс
            0.0,
            thermal_speed,
            size=(self.n, 2),
        )

        # Убираем движение центра масс
        self.velocities -= np.mean(
            self.velocities,
            axis=0,
        )

        # вычисляет реальную температуру
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

        self._apply_outer_walls() # Частицы могут случайно появиться друг на друге. _resolve_collisions() расталкивает их

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

    def pore_intervals(self): # определяет, в каких местах по высоте (по оси Y) в перегородке находятся дырки (поры), через которые могут пролететь частицы

        if self.barrier_mode != 2:
            return []

        pore_h = self.pore_height

        centers = [
            0.20 * WORLD_H,
            0.50 * WORLD_H,
            0.80 * WORLD_H,
        ] # что в перегородке всегда три поры

        intervals = [] # Вычисление границ каждой поры

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

    def obstacles(self): # создает те самые объекты (круги, эллипсы, треугольники), с которыми сталкиваются частицы в специальных режимах (например, «Доска Гальтона» или рассеиватели)
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

        n = self.scatter_count # сколько препятствий должно быть в одной колонке

        # Чтобы соседние препятствия не слипались в сплошную стенку.
        size = min(self.scatter_size_px / FIELD_HEIGHT, 0.25 / n)

        angle = np.radians(self.scatter_angle)

        gradient = self.scatter_shape >= 4 # это проверка на «специальный» режим (например, градиентные шарики, которые уменьшаются в размерах, имитируя доску Гальтона). В коде self.scatter_shape от 4 и выше означает эти режимы
        columns = 5 if gradient else self.scatter_columns
        pitch = (2.4 if gradient else 3.4) * size # расстояние между колонками

        result = []
        k = 0

        for c in range(columns):
            dx = (c - (columns - 1) / 2.0) * pitch # вычисляется смещение колонки относительно центра перегородки

            if c % 2 == 0:
                ys = [(i + 0.5) / n * WORLD_H for i in range(n)]
            else:
                # Сдвинутый ряд, как в доске Гальтона.
                ys = [i / n * WORLD_H for i in range(n + 1)] # создает «шахматный» порядок

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
        bx = self.barrier_x # текущее положение перегородки
        w = self.collision_wall_velocity() # скорость движения перегородки
        r = self.radius

        pos = self.positions
        vel = self.velocities

        if NUMBA_OK:
            obstacles = self.obstacles()
            if self._scatter_kernel_key != self._obstacle_key:
                geometry = np.zeros((len(obstacles), 8))
                vertices = np.zeros((len(obstacles), 3, 2))
                for i, ob in enumerate(obstacles):
                    ellipse = ob.kind == "ellipse"
                    geometry[i] = (ob.dx, ob.cy, ob.cs, ob.sn,
                                   ob.a if ellipse else 0.0, ob.b if ellipse else 0.0,
                                   ob.bound, float(ellipse))
                    if not ellipse:
                        vertices[i] = ob.local
                self._scatter_kernel_data = geometry, vertices
                self._scatter_kernel_key = self._obstacle_key
            scatter_collisions(pos, vel, r, bx, w, *self._scatter_kernel_data)
            return

        for ob in self.obstacles():
            c = ob.center(bx)

            near = np.flatnonzero(
                np.sum((pos - c) ** 2, axis=1) < (ob.bound + r) ** 2
            )

            if len(near) == 0:
                continue

            d, normal = ob.distance(pos[near], bx)

            hit = d < r # проверка: если расстояние меньше радиуса частицы, значит, она врезалась (True)

            if not np.any(hit):
                continue

            idx = near[hit]
            nh = normal[hit]

            pos[idx] += nh * (r - d[hit])[:, None] # Выталкивание из препятствия

            vn = (vel[idx, 0] - w) * nh[:, 0] + vel[idx, 1] * nh[:, 1] # проекция скорости частицы на нормаль

            approaching = vn < 0.0
            j = idx[approaching]

            vel[j] -= 2.0 * vn[approaching][:, None] * nh[approaching] # формула упругого отражения. Мы берем скорость и меняем её нормальную компоненту на противоположную (умножаем на -2 и вычитаем)

    def recolor(self):
        self.colors = self.rng.integers(0, self.color_count, self.n)

    def particle_color_indices(self):
        """Текущая раскраска: постоянные цвета или мгновенная скорость."""
        if self.demon_color_mode == "speed":
            fast = np.linalg.norm(self.velocities, axis=1) > self.demon_color_threshold
            return np.where(fast, 0, 2)  # Красные — быстрые, синие — медленные.
        return self.colors

    def demon_assign_sides(self):
        """Запоминает, по какую сторону линии находится каждая частица."""
        self.demon_side = (self.positions[:, 0] >= self.barrier_x).astype(int)
        self.demon_queue.clear()

    def _demon_place(self, i, side): # аккуратно ставит частицу вплотную к перегородке
        eps = 1e-6
        self.positions[i, 0] = self.barrier_x + (eps if side == 1 else -eps)

    def _demon_reflect(self, i): # Отказ и отскок
        side = int(self.demon_side[i])
        self._demon_place(i, side)

        speed = abs(self.velocities[i, 0])
        self.velocities[i, 0] = speed if side == 1 else -speed

    def _demon_collision(self):
        """
        Удар = центр частицы пересёк линию. Правило по цвету или скорости:
        пропустить, отразить или поставить частицу в очередь на решение.
        Ждущая частица возвращается на свою сторону, скорость не меняется.
        """
        bx = self.barrier_x
        x = self.positions[:, 0]
        side = self.demon_side

        crossed = (
            ((side == 0) & (x >= bx))
            | ((side == 1) & (x < bx))
        ) # те частицы, которые совершили пересечение

        color_indices = self.particle_color_indices() if self.demon_rule_mode == "color" else None
        for i in np.flatnonzero(crossed):
            if self.demon_rule_mode == "speed":
                group = int(np.linalg.norm(self.velocities[i]) > self.demon_speed_threshold)
                mode = self.demon_speed_pass[group]
                otherwise = self.demon_speed_else[group]
            else:
                c = int(color_indices[i])
                mode = self.demon_pass[c]
                otherwise = self.demon_else[c]

            # side == 0: частица летит слева направо (→), side == 1: справа налево (←)
            allowed = (
                mode == 3
                or (mode == 1 and side[i] == 0)
                or (mode == 2 and side[i] == 1)
            ) # разрешено ли этой частице пройти

            if allowed:
                side[i] = 1 - side[i]
            elif otherwise == 1:
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

    def can_pass_array(self, y): # проверяет, может ли частица с заданной координатой Y пролезть сквозь перегородку
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

        return result # Функция возвращает массив «пропусков»

    def particle_can_pass(self, y): # работает только с одной частицей
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

    def _move_barrier(self, dt): # функция вызывается каждый кадр (FPS раз в секунду) и двигает стенку к тому месту, куда вы навели мышку
        """Плавно двигает стенку к цели и запоминает её скорость."""
        minimum, maximum = barrier_motion_limits(self)
        goal = float(np.clip(self.barrier_goal, minimum, maximum))
        self.barrier_goal = goal # куда вы хотите поставить стенку

        move = (goal - self.barrier_fraction) * WORLD_W # это расстояние, которое стенке осталось проехать до цели
        move *= min(1.0, dt * BARRIER_RESPONSE) # Затем мы умножаем это расстояние на коэффициент BARRIER_RESPONSE (скорость реакции)

        limit = BARRIER_MAX_SPEED * dt
        move = float(np.clip(move, -limit, limit))

        self.barrier_fraction += move / WORLD_W # обновляется реальное положение стенки
        self.barrier_velocity = move / dt # скорость стенки

    def _barrier_collision(self, old_x, old_bx): # решает, как частицы должны сталкиваться с перегородкой, в зависимости от выбранного вами режима
        """
        Столкновение с бесконечно тяжёлой подвижной стенкой.

        Сторону определяем по положению ДО шага, чтобы быстрая стенка
        не могла «проскочить» сквозь частицу.

        В системе отсчёта стенки нормальная компонента скорости
        меняет знак: v' - w = -(v - w)  =>  v' = 2w - v.
        """
        if self.barrier_mode == 0: # перегородки нет
            return

        if self.barrier_mode == 3: # рассеиватели (круги, треугольники)
            self._collide_scatterers()
            return

        if self.barrier_mode == 4: # полупроницаемая стенка
            self._semipermeable_crossing(old_x, old_bx)
            return

        if self.barrier_mode == 5: # Демон Максвелла
            self._demon_collision()
            return

        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        blocked = ~self.can_pass_array(self.positions[:, 1]) # массив «кто не может пройти»

        x = self.positions[:, 0]
        v = self.velocities[:, 0]

        left = blocked & (old_x < old_bx) # частицы, которые были слева и не могут пройти
        right = blocked & (old_x >= old_bx)

        hit = left & (x > bx - r) # частицы, которые были слева и в этом кадре «залезли» на стенку
        self.positions[hit, 0] = bx - r
        bounce = hit & (v > w) # проверяем, движется ли частица навстречу стенке
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

        # для сплошной стенки (1) и стенки с порами (2)

        bx = self.barrier_x
        w = self.collision_wall_velocity()
        r = self.radius

        blocked = ~self.can_pass_array(self.positions[:, 1])

        x = self.positions[:, 0]
        v = self.velocities[:, 0]

        near = blocked & (np.abs(x - bx) < r) # частицы, которые в результате столкновений оказались внутри перегородки (застряли в ней)

        left = near & (x < bx)
        self.positions[left, 0] = bx - r
        bounce = left & (v > w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0] # Выталкивание влево

        right = near & (x >= bx)
        self.positions[right, 0] = bx + r
        bounce = right & (v < w)
        self.velocities[bounce, 0] = 2.0 * w - self.velocities[bounce, 0]

    # --------------------------------------------------------
    # Стенки
    # --------------------------------------------------------

    def _apply_outer_walls(self): # гарантирует, что ни одна частица не сможет вылететь за пределы экрана
        if len(self.positions) == 0:
            return

        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have matching shapes")
        if NUMBA_OK:
            outer_walls(self.positions, self.velocities, self.radius, WORLD_W, WORLD_H)
            return
        r = self.radius

        # Левая стенка
        mask = self.positions[:, 0] < r # слишком близко подлетели к левому краю
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

    def _resolve_collisions(self): # каким способом рассчитывать столкновения частиц друг с другом
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

        cell = 2.0 * self.radius * 1.05
        cells = (int(WORLD_W / cell) + 3) * (int(WORLD_H / cell) + 3)
        count = len(pos)
        if pos.shape != vel.shape:
            raise ValueError("positions and velocities must have matching shapes")
        key = (count, cells)
        if self._collision_workspace is None or self._collision_workspace[0] != key:
            self._collision_workspace = (key, np.empty(cells, dtype=np.int64),
                                         *(np.empty(count, dtype=np.int64) for _ in range(3)))
        collide_numba(pos, vel, self.radius, WORLD_W, WORLD_H,
                      *self._collision_workspace[1:])

        self._apply_outer_walls() # возвращает их в рамки
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
    # Ланжевеновская динамика (статистическая симуляция, но на практике он заменил сложные силы трения и шума на более простые и быстрые упругие столкновения)
    # --------------------------------------------------------

    def step(self, dt): # вызывается по таймеру FPS (каждый кадр анимации) и описывает, как вся система делает один маленький шаг во времени
        if not self.potential_defined:
            return

        # Контролируем K + U, а не температуру: обмен между K и U остаётся.
        if not np.isfinite(dt) or not 0.0 < dt <= DT * (1.0 + 1e-10):
            raise ValueError("dt: expected a positive integration step no larger than DT")
        self._validate_configuration()
        accelerated = (NUMBA_OK and self.positions.dtype == np.float64
                       and self.velocities.dtype == np.float64
                       and self.positions.flags.c_contiguous and self.velocities.flags.c_contiguous)
        energy_before = 0.0 if accelerated else self.mechanical_energy()
        old_bx = self.barrier_x # Движение перегородки
        if self.demon_mode:
            # Линия неподвижна и ставится строго по координате.
            self.barrier_fraction = float(
                np.clip(self.barrier_goal, BARRIER_MIN, BARRIER_MAX)
            )
            self.barrier_velocity = 0.0
        else:
            self._move_barrier(dt)

        if accelerated:
            if self._old_x.shape != (self.n,):
                self._old_x = np.empty(self.n)
            old_x = self._old_x
            energy_before = kick_drift(self.positions, self.velocities, old_x,
                                       self.potential, self.gradient, self.energy_scale, WORLD_W, dt)
        else:
            force = self.force_x(self.positions[:, 0])
            self.velocities[:, 0] += 0.5 * dt * force
            old_x = self.positions[:, 0].copy()
            self.positions += self.velocities * dt

        self._apply_outer_walls()
        moving_wall = self.barrier_mode in (1, 2, 4) and self.collision_wall_velocity() != 0.0
        kinetic_before_wall = self.kinetic_energy() if moving_wall else 0.0
        self._barrier_collision(old_x, old_bx)
        wall_work = self.kinetic_energy() - kinetic_before_wall if moving_wall else 0.0

        if accelerated:
            finish_kick(self.positions, self.velocities, self.gradient, self.energy_scale, WORLD_W, dt)
        else:
            self.velocities[:, 0] += 0.5 * dt * self.force_x(self.positions[:, 0])

        # Раздвигание перекрывающихся дисков тоже меняет U(x).
        kinetic_before_collisions = self.kinetic_energy() if moving_wall else 0.0
        self._resolve_collisions()
        if moving_wall:
            wall_work += self.kinetic_energy() - kinetic_before_collisions
        if accelerated:
            temperature = stabilize_energy(self.positions, self.velocities, self.potential,
                                           self.energy_scale, WORLD_W, energy_before + wall_work)
            self.check_temperature_limit(temperature)
        else:
            self._stabilize_energy(energy_before + wall_work)
            self.check_temperature_limit()

    def _validate_configuration(self):
        if self.positions.shape != (self.n, 2) or self.velocities.shape != (self.n, 2):
            raise ValueError("particle arrays do not match the particle count")
        signature = tuple(getattr(self, name) for name in PARAMETER_RANGES)
        signature += (tuple(self.demon_pass), tuple(self.demon_else),
                      tuple(self.demon_speed_pass), tuple(self.demon_speed_else),
                      self.demon_rule_mode, self.demon_color_mode)
        if signature != self._validated_configuration:
            validate_parameters(self)
            self._validated_configuration = signature

    def kinetic_energy(self):
        return float(0.5 * np.sum(self.velocities ** 2))

    def mechanical_energy(self):
        """Полная механическая энергия газа, массы молекул равны единице."""
        if NUMBA_OK and self.potential_defined:
            k, u = energy_totals(self.positions, self.velocities, self.potential, self.energy_scale, WORLD_W)
            return float(k + u)
        return self.kinetic_energy() + self.n * self.mean_potential_energy()

    def _stabilize_energy(self, target_energy):
        """Проекция на баланс энергии после дискретного шага и столкновений.

        Это численная коррекция замкнутой модели, не термостат: заданная
        температура не используется, физическая работа стенки учтена в цели.
        Направления и отношения модулей скоростей сохраняются.
        """
        kinetic = self.kinetic_energy()
        potential = self.n * self.mean_potential_energy()
        if not np.isfinite(kinetic + potential + target_energy):
            return  # Проверка температуры сообщит о нечисловом состоянии.
        desired = max(0.0, target_energy - potential)
        tolerance = 1e-12 * max(1.0, abs(target_energy), abs(potential), kinetic)
        if kinetic > 0.0 and abs(desired - kinetic) > tolerance:
            self.velocities *= np.sqrt(desired / kinetic)

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

        counts, edges = np.histogram( # counts — это массив из 40 чисел. В каждом числе записано, сколько частиц попало в эту корзину (например, [2, 5, 12, 30, 15...])
            normalized_x,
            bins=bins,
            range=(0.0, 1.0),
        )

        probabilities = (
            counts.astype(float) / self.n
        )

        return probabilities, edges

    def theoretical_distribution(self, bins=HIST_BINS): # рисует красную линию (теорию Больцмана) на нижнем графике.
        key = (bins, self._potential_revision, self.temperature, self.energy_scale)
        if key in self._theory_cache:
            return self._theory_cache[key]
        centers = (
            np.arange(bins) + 0.5
        ) / bins

        potential = np.interp( # Получение потенциальной энергии в этих точках
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
        energy -= np.min(energy) # Мы сдвигаем всю кривую так, чтобы её минимум стал равен нулю. Это нужно, чтобы в следующей формуле экспонента не выдала бесконечно большое число.

        weights = np.exp(
            -np.clip(
                energy / T,
                0.0,
                700.0,
            )
        )

        # Reference equilibrium ignores partitions; their effects appear in the histogram.
        total = np.sum(weights)

        if total <= 0:
            weights[:] = 1.0
            total = np.sum(weights)

        result = centers, weights / total
        for array in result:
            array.setflags(write=False)
        self._theory_cache = {key: result}
        return result

    def center_of_mass(self):
        if self.n == 0:
            return 0.0

        return float(
            np.mean(self.positions[:, 0])
            / WORLD_W
        )

    def entropy(self, bins=HIST_BINS):
        """
        Дискретная крупнозернистая энтропия:
            S = -sum p_i ln(p_i)

        k_B = 1.
        """

        p, _ = self.histogram(bins=bins)

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

    def mean_potential_energy(self):
        """Вычисляет среднюю потенциальную энергию всех частиц."""
        if self.n == 0 or not self.potential_defined:
            return 0.0
        
        # Находим энергию U(x) для каждой частицы по её координате X
        if NUMBA_OK:
            return float(energy_totals(self.positions, self.velocities, self.potential,
                                      self.energy_scale, WORLD_W)[1] / self.n)
        energies = np.interp(
            self.positions[:, 0],
            self.potential_x,
            self.potential * self.energy_scale
        )
        return float(np.mean(energies))

    def free_energy(self, bins=HIST_BINS):
        """Вычисляет свободную энергию Гельмгольца: F = <U> - T*S."""
        if self.n == 0:
            return 0.0
            
        mean_u = self.mean_potential_energy()
        entropy = self.entropy(bins=bins)
        temp = self.temperature
        
        return float(mean_u - temp * entropy)

    def check_temperature_limit(self, temperature=None):
        if temperature is None:
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
    def statistics(self, bins=HIST_BINS):
        """One histogram and one energy pass for all readings at this instant."""
        probabilities, _ = self.histogram(bins)
        positive = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        if NUMBA_OK:
            kinetic, potential = energy_totals(self.positions, self.velocities, self.potential,
                                                self.energy_scale, WORLD_W)
            mean_u = potential / self.n if self.potential_defined else 0.0
            temperature = kinetic / self.n
        else:
            mean_u = self.mean_potential_energy()
            temperature = self.effective_temperature()
        # F retains the original fixed 40-bin definition, independent of display bins.
        free_entropy = entropy if bins == HIST_BINS else self.entropy()
        return probabilities, entropy, temperature, mean_u, mean_u - self.temperature * free_entropy
