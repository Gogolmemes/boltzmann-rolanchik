import numpy as np

try:
    from numba import njit
    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(cache=True)
def collide_numba(pos, vel, radius, world_w, world_h,
                  head=None, nxt=None, cxs=None, cys=None):
    """
    Упругие столкновения одинаковых дисков.
    Сетка на связных списках, без выделения памяти внутри цикла по парам.
    Меняет pos и vel на месте.
    """
    n = pos.shape[0]
    d = 2.0 * radius
    d2 = d * d
    cell = d * 1.05

    nx = int(world_w / cell) + 3
    ny = int(world_h / cell) + 3

    if head is None:
        head = np.empty(nx * ny, np.int64)
        nxt = np.empty(n, np.int64)
        cxs = np.empty(n, np.int64)
        cys = np.empty(n, np.int64)
    head.fill(-1)

    for i in range(n):
        cx = min(max(int(pos[i, 0] / cell), 0), nx - 1)
        cy = min(max(int(pos[i, 1] / cell), 0), ny - 1)
        cxs[i] = cx
        cys[i] = cy
        c = cy * nx + cx
        nxt[i] = head[c]
        head[c] = i

    for i in range(n):
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                x2 = cxs[i] + ox
                y2 = cys[i] + oy

                if x2 < 0 or y2 < 0 or x2 >= nx or y2 >= ny:
                    continue

                j = head[y2 * nx + x2]

                while j != -1:
                    if j > i:
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        dist2 = dx * dx + dy * dy

                        if 1e-16 < dist2 < d2:
                            dist = np.sqrt(dist2)
                            nxn = dx / dist
                            nyn = dy / dist
                            half = 0.5 * (d - dist)

                            pos[i, 0] -= nxn * half
                            pos[i, 1] -= nyn * half
                            pos[j, 0] += nxn * half
                            pos[j, 1] += nyn * half

                            rel = (
                                (vel[j, 0] - vel[i, 0]) * nxn
                                + (vel[j, 1] - vel[i, 1]) * nyn
                            )

                            if rel < 0.0:
                                vel[i, 0] += rel * nxn
                                vel[i, 1] += rel * nyn
                                vel[j, 0] -= rel * nxn
                                vel[j, 1] -= rel * nyn

                    j = nxt[j]



@njit(cache=True, inline="always")
def sample_uniform(values, x, world_w):
    """Linear interpolation on the potential's uniform grid, including endpoints."""
    t = min(max(x * ((len(values) - 1) / world_w), 0.0), len(values) - 1.0)
    i = min(int(t), len(values) - 2)
    return values[i] + (t - i) * (values[i + 1] - values[i])


@njit(cache=True)
def energy_totals(pos, vel, potential, scale, world_w):
    kinetic = 0.0
    energy = 0.0
    for i in range(len(pos)):
        kinetic += 0.5 * (vel[i, 0]**2 + vel[i, 1]**2)
        energy += scale * sample_uniform(potential, pos[i, 0], world_w)
    return kinetic, energy


@njit(cache=True)
def kick_drift(pos, vel, old_x, potential, gradient, scale, world_w, dt):
    """Energy before the step, the first Verlet kick and drift in one pass."""
    energy = 0.0
    for i in range(len(pos)):
        x = pos[i, 0]
        old_x[i] = x
        energy += 0.5 * (vel[i, 0]**2 + vel[i, 1]**2)
        energy += scale * sample_uniform(potential, x, world_w)
        vel[i, 0] -= 0.5 * dt * scale * sample_uniform(gradient, x, world_w)
        pos[i, 0] += dt * vel[i, 0]
        pos[i, 1] += dt * vel[i, 1]
    return energy


@njit(cache=True)
def finish_kick(pos, vel, gradient, scale, world_w, dt):
    for i in range(len(pos)):
        vel[i, 0] -= 0.5 * dt * scale * sample_uniform(gradient, pos[i, 0], world_w)


@njit(cache=True)
def outer_walls(pos, vel, radius, world_w, world_h):
    for i in range(len(pos)):
        for axis in range(2):
            upper = (world_w if axis == 0 else world_h) - radius
            if pos[i, axis] < radius:
                pos[i, axis] = radius
                vel[i, axis] = abs(vel[i, axis])
            if pos[i, axis] > upper:
                pos[i, axis] = upper
                vel[i, axis] = -abs(vel[i, axis])


@njit(cache=True)
def stabilize_energy(pos, vel, potential, scale, world_w, target):
    kinetic, energy = energy_totals(pos, vel, potential, scale, world_w)
    if not np.isfinite(kinetic + energy + target):
        return np.nan
    desired = max(0.0, target - energy)
    tolerance = 1e-12 * max(1.0, abs(target), abs(energy), kinetic)
    if kinetic > 0.0 and abs(desired - kinetic) > tolerance:
        factor = np.sqrt(desired / kinetic)
        for i in range(len(vel)):
            vel[i, 0] *= factor
            vel[i, 1] *= factor
        kinetic = desired
    return kinetic / max(1, len(pos))



@njit(cache=True)
def scatter_collisions(pos, vel, radius, bx, wall_speed, geometry, vertices):
    """Same ellipse approximation and triangle normals as Obstacle.distance."""
    for obstacle in range(len(geometry)):
        dx, cy, cs, sn, a, b, bound, ellipse = geometry[obstacle]
        cx = bx + dx
        reach2 = (bound + radius)**2
        for i in range(len(pos)):
            px, py = pos[i, 0] - cx, pos[i, 1] - cy
            if px * px + py * py >= reach2:
                continue
            if ellipse:
                u, v = cs * px + sn * py, -sn * px + cs * py
                f = max(np.sqrt((u / a)**2 + (v / b)**2), 1e-9)
                gu, gv = u / (a * a * f), v / (b * b * f)
                g = np.sqrt(gu * gu + gv * gv) + 1e-12
                distance = (f - 1.0) / g
                nx, ny = cs * gu / g - sn * gv / g, sn * gu / g + cs * gv / g
            else:
                best = np.inf
                nx = ny = edge_x = edge_y = 0.0
                inside = True
                for k in range(3):
                    # Absolute vertices match the reference arithmetic near a moving wall.
                    ax = cx + vertices[obstacle, k, 0]
                    ay = cy + vertices[obstacle, k, 1]
                    ex = cx + vertices[obstacle, (k + 1) % 3, 0] - ax
                    ey = cy + vertices[obstacle, (k + 1) % 3, 1] - ay
                    length2 = ex * ex + ey * ey
                    inv_length = 1.0 / np.sqrt(length2)
                    enx, eny = ey * inv_length, -ex * inv_length
                    rx, ry = pos[i, 0] - ax, pos[i, 1] - ay
                    inside = inside and rx * enx + ry * eny <= 0.0
                    t = min(max((rx * ex + ry * ey) / length2, 0.0), 1.0)
                    qx, qy = ax + t * ex, ay + t * ey
                    vx, vy = pos[i, 0] - qx, pos[i, 1] - qy
                    dist = np.sqrt(vx * vx + vy * vy)
                    if dist < best:
                        best = dist
                        divisor = max(dist, 1e-12)
                        nx, ny = vx / divisor, vy / divisor
                        edge_x, edge_y = enx, eny
                distance = -best if inside else best
                if inside:
                    nx, ny = edge_x, edge_y
            if distance < radius:
                pos[i, 0] += nx * (radius - distance)
                pos[i, 1] += ny * (radius - distance)
                normal_speed = (vel[i, 0] - wall_speed) * nx + vel[i, 1] * ny
                if normal_speed < 0.0:
                    vel[i, 0] -= 2.0 * normal_speed * nx
                    vel[i, 1] -= 2.0 * normal_speed * ny


def warmup_numba():
    """Компиляция ядра на крошечном массиве (вызывается при старте)."""
    if not NUMBA_OK:
        return

    pos = np.random.rand(8, 2)
    vel = np.random.randn(8, 2)
    n = len(pos)
    head = np.empty(10000, dtype=np.int64)
    work = [np.empty(n, dtype=np.int64) for _ in range(3)]
    collide_numba(pos, vel, 0.01, 3.2, 1.0, head, *work)
    potential = np.zeros(1024)
    old_x = np.empty(n)
    target = kick_drift(pos, vel, old_x, potential, potential, 1.0, 3.2, 0.001)
    outer_walls(pos, vel, 0.01, 3.2, 1.0)
    finish_kick(pos, vel, potential, 1.0, 3.2, 0.001)
    stabilize_energy(pos, vel, potential, 1.0, 3.2, target)
    geometry = np.array([[0.0, 0.5, 1.0, 0.0, 0.1, 0.1, 0.1, 1.0]])
    scatter_collisions(pos, vel, 0.01, 1.6, 0.0, geometry, np.zeros((1, 3, 2)))

class Obstacle:
    """
    Выпуклое препятствие, жёстко привязанное к перегородке.
    kind = "ellipse" (круг, если a == b) или "triangle".
    Координаты dx, cy: смещение по x от перегородки и y в мировых единицах.
    """

    def __init__(self, kind, dx, cy, angle, a=0.0, b=0.0, radius=0.0):
        self.kind = kind
        self.dx = dx
        self.cy = cy
        self.cs = float(np.cos(angle))
        self.sn = float(np.sin(angle))

        if kind == "ellipse":
            self.a = a
            self.b = b
            self.bound = max(a, b)
        else:
            t = angle + np.arange(3) * 2.0 * np.pi / 3.0
            self.local = radius * np.column_stack((np.cos(t), np.sin(t)))
            self.bound = radius

    def center(self, bx):
        return np.array([bx + self.dx, self.cy])

    def distance(self, p, bx):
        """Знаковое расстояние от центров частиц до поверхности и внешняя нормаль."""
        c = self.center(bx)

        if self.kind == "ellipse":
            return self._ellipse_distance(p, c)

        return self._polygon_distance(p, c + self.local)

    def _ellipse_distance(self, p, c):
        d = p - c
        u = self.cs * d[:, 0] + self.sn * d[:, 1]
        v = -self.sn * d[:, 0] + self.cs * d[:, 1]

        f = np.maximum(np.sqrt((u / self.a) ** 2 + (v / self.b) ** 2), 1e-9)

        gu = u / (self.a ** 2 * f)
        gv = v / (self.b ** 2 * f)
        g = np.sqrt(gu ** 2 + gv ** 2) + 1e-12

        # Расстояние в первом порядке: (F - 1) / |grad F|.
        dist = (f - 1.0) / g

        nu = gu / g
        nv = gv / g

        normal = np.column_stack((
            self.cs * nu - self.sn * nv,
            self.sn * nu + self.cs * nv,
        ))

        return dist, normal

    @staticmethod
    def _polygon_distance(p, verts):
        count = len(verts)

        best = np.full(len(p), np.inf)
        best_normal = np.zeros((len(p), 2))
        best_edge_normal = np.zeros((len(p), 2))
        inside = np.ones(len(p), dtype=bool)

        for k in range(count):
            a = verts[k]
            b = verts[(k + 1) % count]

            e = b - a
            length2 = float(e @ e)
            edge_normal = np.array([e[1], -e[0]]) / np.sqrt(length2)

            t = np.clip(((p - a) @ e) / length2, 0.0, 1.0)
            q = a + t[:, None] * e

            diff = p - q
            dist = np.linalg.norm(diff, axis=1)

            inside &= ((p - a) @ edge_normal) <= 0.0

            better = dist < best
            best[better] = dist[better]
            best_normal[better] = (
                diff[better] / np.maximum(dist[better], 1e-12)[:, None]
            )
            best_edge_normal[better] = edge_normal

        d = np.where(inside, -best, best)
        n = np.where(inside[:, None], best_edge_normal, best_normal)

        return d, n

    def outline(self, bx, points=36):
        """Контур для рисования (в мировых координатах)."""
        c = self.center(bx)

        if self.kind == "ellipse":
            t = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
            lx = self.a * np.cos(t)
            ly = self.b * np.sin(t)

            return np.column_stack((
                c[0] + self.cs * lx - self.sn * ly,
                c[1] + self.sn * lx + self.cs * ly,
            ))

        return c + self.local