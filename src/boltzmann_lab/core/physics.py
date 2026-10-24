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
def collide_numba(pos, vel, radius, world_w, world_h):
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

    head = np.full(nx * ny, -1, np.int64)
    nxt = np.full(n, -1, np.int64)
    cxs = np.empty(n, np.int64)
    cys = np.empty(n, np.int64)

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


def warmup_numba():
    """Компиляция ядра на крошечном массиве (вызывается при старте)."""
    if not NUMBA_OK:
        return

    pos = np.random.rand(8, 2)
    vel = np.random.randn(8, 2)
    collide_numba(pos, vel, 0.01, 3.2, 1.0)

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