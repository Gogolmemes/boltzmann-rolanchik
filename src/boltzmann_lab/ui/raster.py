"""Draw the Pygame disk masks in one compiled call, without Python per particle."""
import numpy as np
from boltzmann_lab.core.physics import njit


@njit(cache=True)
def paint_disks(pixels, xs, ys, colors, dx, dy, border, white):
    width, height = pixels.shape
    for i in range(len(xs)):
        for j in range(len(dx)):
            x, y = xs[i] + dx[j], ys[i] + dy[j]
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = white if border[j] else colors[i]


def warmup_raster():
    pixels = np.zeros((4, 4), dtype=np.uint32, order="F")
    points = np.array([1], dtype=np.int64)
    offsets = np.array([0], dtype=np.int64)
    paint_disks(pixels, points, points, np.array([1], dtype=np.uint32),
                offsets, offsets, np.array([False]), 0)
