import numpy as np
import pygame

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

        pxs = (sim.positions[:, 0] * (FIELD_WIDTH / WORLD_W)).astype(int).tolist()
        pys = (sim.positions[:, 1] * (FIELD_HEIGHT / WORLD_H)).astype(int).tolist()

        if sim.demon_mode and len(sim.colors) == sim.n:
            colors = [DEMON_COLORS[c] for c in sim.colors.tolist()]
        else:
            speeds = np.linalg.norm(sim.velocities, axis=1)
            ratio = np.clip(speeds / reference_speed, 0.0, 1.0)

            red = (60 + 195 * ratio).astype(int)
            green = np.minimum(255, (180 + 60 * ratio).astype(int))
            blue = np.maximum(0, (255 - 190 * ratio).astype(int))

            colors = list(zip(red.tolist(), green.tolist(), blue.tolist()))

        radius_px = sim.radius_px

        for px, py, color in zip(pxs, pys, colors):
            pygame.draw.circle(surface, color, (px, py), radius_px)
            pygame.draw.circle(surface, (235, 245, 255), (px, py), radius_px, 1)

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

        rgb = pygame.surfarray.array3d(
            surface
        )

        # pygame: [x, y, rgb]
        # Qt:     [y, x, rgb]
        rgb = np.transpose(
            rgb,
            (1, 0, 2),
        )

        rgb = np.ascontiguousarray(rgb)

        image = QImage(
            rgb.data,
            FIELD_WIDTH,
            FIELD_HEIGHT,
            FIELD_WIDTH * 3,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)

        target_size = (
            self.particle_view.size()
        )

        pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.particle_view.setPixmap(
            pixmap
        )

    def _draw_obstacles(self, surface, bx):
        # Тонкая направляющая: за неё удобно хватать перегородку.
        pygame.draw.line(
            surface, (90, 60, 70), (bx, 0), (bx, FIELD_HEIGHT), 1
        )

        scale = FIELD_HEIGHT / WORLD_H

        for ob in self.sim.obstacles():
            points = [
                (int(x * scale), int(y * scale))
                for x, y in ob.outline(self.sim.barrier_x)
            ]

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