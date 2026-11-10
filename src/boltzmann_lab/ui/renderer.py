import numpy as np
import pygame

from boltzmann_lab.core.physics import NUMBA_OK
from boltzmann_lab.ui.raster import paint_disks
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt #######################################

from boltzmann_lab.core.config import (
    FIELD_WIDTH,
    FIELD_HEIGHT,
    WORLD_H,
    WORLD_W,
    POTENTIAL_POINTS,
    DEMON_COLORS,
    DEMON_HEX,
)


class ParticleRendererMixin:
    """Методы отрисовки поля частиц для MainWindow."""
    def render_particle_field(self):
        surface = self.surface

        surface.fill((5, 10, 22))

        # ----------------------------------------------------
        # Фон показывает потенциальную энергию
        # ----------------------------------------------------

        if self.potential_defined:
            if self._bg_surface is None:
                self._bg_surface = self._build_background()

            surface.blit(self._bg_surface, (0, 0))

        # ----------------------------------------------------
        # Сетка
        # ----------------------------------------------------

        grid_color = (35, 48, 70)

        for i in range(1, 10):
            x = int(
                FIELD_WIDTH * i / 10
            )

            pygame.draw.line(
                surface,
                grid_color,
                (x, 0),
                (x, FIELD_HEIGHT),
                1,
            )

        for i in range(1, 4):
            y = int(
                FIELD_HEIGHT * i / 4
            )

            pygame.draw.line(
                surface,
                grid_color,
                (0, y),
                (FIELD_WIDTH, y),
                1,
            )

        # ----------------------------------------------------
        # Перегородка
        # ----------------------------------------------------

        if self.sim.barrier_mode != 0:
            mode = self.sim.barrier_mode

            bx = int(
                self.sim.barrier_fraction * FIELD_WIDTH
            )

            barrier_color = (248, 113, 113)

            if mode == 1:
                pygame.draw.line(
                    surface,
                    barrier_color,
                    (bx, 0),
                    (bx, FIELD_HEIGHT),
                    5,
                )

            elif mode == 2:
                intervals = self.sim.pore_intervals()

                pixel_intervals = [
                    (int(a * FIELD_HEIGHT), int(b * FIELD_HEIGHT))
                    for a, b in intervals
                ]

                current = 0

                for a, b in pixel_intervals:
                    if a > current:
                        pygame.draw.line(
                            surface, barrier_color,
                            (bx, current), (bx, a), 5,
                        )

                    current = max(current, b)

                if current < FIELD_HEIGHT:
                    pygame.draw.line(
                        surface, barrier_color,
                        (bx, current), (bx, FIELD_HEIGHT), 5,
                    )

            elif mode == 3:
                self._draw_obstacles(surface, bx)
            elif mode == 5:
                self._draw_demon(surface, bx)
            else:
                self._draw_semipermeable(surface, bx)

        # ----------------------------------------------------
        # Центр масс
        # ----------------------------------------------------

        if self.sim.n > 0:
            com = self.sim.center_of_mass()

            com_x = int(
                com * FIELD_WIDTH
            )

            pygame.draw.line(
                surface,
                (250, 204, 21),
                (com_x, 0),
                (com_x, FIELD_HEIGHT),
                2,
            )

        # ----------------------------------------------------
        # Частицы
        # ----------------------------------------------------

        sim = self.sim

        T = max(sim.temperature, 1e-6)
        reference_speed = 3.0 * np.sqrt(T)

        pxs = (sim.positions[:, 0] * (FIELD_WIDTH / WORLD_W)).astype(np.int64)
        pys = (sim.positions[:, 1] * (FIELD_HEIGHT / WORLD_H)).astype(np.int64)

        if sim.demon_mode and len(sim.colors) == sim.n:
            colors = np.asarray(DEMON_COLORS, dtype=np.uint32)[sim.particle_color_indices()]
        else:
            speeds = np.linalg.norm(sim.velocities, axis=1)
            ratio = np.clip(speeds / reference_speed, 0.0, 1.0)

            red = (60 + 195 * ratio).astype(int)
            green = np.minimum(255, (180 + 60 * ratio).astype(int))
            blue = np.maximum(0, (255 - 190 * ratio).astype(int))

            colors = np.column_stack((red, green, blue)).astype(np.uint32)

        # Games may magnify markers for a projector without changing collision radii.
        radius_px = getattr(self, "particle_draw_radius_px", sim.radius_px)

        self._paint_disks(surface, pxs, pys, colors, radius_px)

        if self.sim.demon_mode and self.sim.demon_queue:
            i = self.sim.demon_queue[0]

            px = int(self.sim.positions[i, 0] / WORLD_W * FIELD_WIDTH)
            py = int(self.sim.positions[i, 1] / WORLD_H * FIELD_HEIGHT)

            pygame.draw.circle(
                surface, (250, 204, 21),
                (px, py), self.sim.radius_px + 7, 3,
            )

        # Рамка
        pygame.draw.rect(
            surface,
            (100, 116, 139),
            (
                0,
                0,
                FIELD_WIDTH - 1,
                FIELD_HEIGHT - 1,
            ),
            2,
        )

        # ----------------------------------------------------
        # Pygame Surface -> NumPy -> QImage -> QLabel
        # ----------------------------------------------------

        # Convert directly to row-major RGB, without a transposed NumPy copy.
        rgb = pygame.image.tobytes(surface, "RGB")
        image = QImage(rgb, FIELD_WIDTH, FIELD_HEIGHT, FIELD_WIDTH * 3,
                       QImage.Format.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(image)

        if not self.particle_view.hasHeightForWidth():
            pixmap = pixmap.scaled(
                self.particle_view.contentsRect().size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.particle_view.setPixmap(
            pixmap
        )

    def _paint_disks(self, surface, xs, ys, colors, radius):
        if not NUMBA_OK or surface.get_bitsize() != 32:
            surface.lock()
            try:
                for x, y, color in zip(xs.tolist(), ys.tolist(), colors.tolist()):
                    pygame.draw.circle(surface, color, (x, y), radius)
                    pygame.draw.circle(surface, (235, 245, 255), (x, y), radius, 1)
            finally:
                surface.unlock()
            return
        if getattr(self, "_disk_mask_radius", None) != radius:
            marker = pygame.Surface((2 * radius + 1, 2 * radius + 1), depth=32)
            marker.fill(0)
            pygame.draw.circle(marker, (1, 0, 0), (radius, radius), radius)
            pygame.draw.circle(marker, (2, 0, 0), (radius, radius), radius, 1)
            mask = pygame.surfarray.array2d(marker)
            x, y = np.nonzero(mask)
            border = mask[x, y] == marker.map_rgb((2, 0, 0))
            self._disk_mask = ((x - radius).astype(np.int64),
                               (y - radius).astype(np.int64), border)
            self._disk_mask_radius = radius
        shifts = surface.get_shifts()
        packed = ((colors[:, 0] << shifts[0]) | (colors[:, 1] << shifts[1])
                  | (colors[:, 2] << shifts[2])).astype(np.uint32)
        pixels = pygame.surfarray.pixels2d(surface)
        try:
            paint_disks(pixels, xs, ys, packed, *self._disk_mask,
                        surface.map_rgb((235, 245, 255)))
        finally:
            del pixels

    def _draw_obstacles(self, surface, bx):
        # Тонкая направляющая: за неё удобно хватать перегородку.
        pygame.draw.line(
            surface, (90, 60, 70), (bx, 0), (bx, FIELD_HEIGHT), 1
        )

        scale = FIELD_HEIGHT / WORLD_H

        obstacles = self.sim.obstacles()
        key = self.sim._obstacle_key, self.sim.barrier_x
        if getattr(self, "_obstacle_polygons_key", None) != key:
            self._obstacle_polygons = [
                [(int(x * scale), int(y * scale)) for x, y in ob.outline(self.sim.barrier_x)]
                for ob in obstacles
            ]
            self._obstacle_polygons_key = key
        for points in self._obstacle_polygons:
            pygame.draw.polygon(surface, (120, 50, 60), points)
            pygame.draw.polygon(surface, (248, 113, 113), points, 2)

    def _draw_semipermeable(self, surface, bx):
        base = (167, 139, 250)

        for y in range(0, FIELD_HEIGHT, 14):
            pygame.draw.line(
                surface, base,
                (bx, y), (bx, min(y + 8, FIELD_HEIGHT)), 4,
            )

        def shade(p):
            return tuple(int(40 + (c - 40) * p) for c in base)

        # Стрелки: яркость пропорциональна вероятности прохода.
        for fy in (0.2, 0.5, 0.8):
            y = int(fy * FIELD_HEIGHT)

            pygame.draw.polygon(
                surface, shade(self.sim.p_lr),
                [(bx - 20, y - 7), (bx - 20, y + 7), (bx - 8, y)],
            )

            pygame.draw.polygon(
                surface, shade(self.sim.p_rl),
                [(bx + 20, y - 7), (bx + 20, y + 7), (bx + 8, y)],
            )

    def _draw_demon(self, surface, bx):
        for y in range(0, FIELD_HEIGHT, 10):
            pygame.draw.line(
                surface, (226, 232, 240),
                (bx, y), (bx, min(y + 5, FIELD_HEIGHT)), 3,
            )

    def _build_background(self):
        bg = pygame.Surface((FIELD_WIDTH, FIELD_HEIGHT))
        bg.fill((5, 10, 22))

        strips = 150

        s = (np.arange(strips) + 0.5) / strips

        u = np.interp(
            s,
            np.linspace(0.0, 1.0, POTENTIAL_POINTS),
            self.sim.potential,
        )
        u = np.clip(u, 0.0, 1.0)

        for i in range(strips):
            x0 = int(i * FIELD_WIDTH / strips)
            x1 = int((i + 1) * FIELD_WIDTH / strips)
            ui = float(u[i])

            color = (
                int(8 + 50 * ui),
                int(18 + 12 * ui),
                int(45 + 35 * (1.0 - ui)),
            )

            pygame.draw.rect(
                bg, color, (x0, 0, max(1, x1 - x0), FIELD_HEIGHT)
            )

        return bg