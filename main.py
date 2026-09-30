#!/usr/bin/env python3
"""
Boltzmann Lab — interactive 1D potential and Langevin-particle simulator.

Single-file, standard-library-only application (tkinter, math, random, json).
Designed for Windows and Linux with Python 3.10+.

Model units:
    mass m = 1
    Boltzmann constant k_B = 1
    x in [0, 1]
    U(x) = energy_scale * u_drawn(x), where u_drawn is normalized to [0, 1]

The underdamped Langevin equation is integrated with a BAOAB-like splitting:
    dx = v dt
    dv = F(x) dt - gamma v dt + sqrt(2 gamma T) dW
    F(x) = -dU/dx

The particle display is a 2D strip so that finite-radius elastic collisions and
porous partitions can be visualized. The external potential depends only on x,
so the target ideal-gas x-distribution is proportional to exp[-U(x)/T].
"""

from __future__ import annotations

import json
import math
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Optional


# -----------------------------------------------------------------------------
# Localization
# -----------------------------------------------------------------------------

TEXT = {
    "ru": {
        "app_title": "Boltzmann Lab — распределение Больцмана",
        "potential_title": "Потенциальная энергия U(x): рисуйте мышью",
        "particles_title": "Движение частиц",
        "distribution_title": "Распределение по x",
        "controls": "Параметры модели",
        "language": "Язык",
        "russian": "Русский",
        "english": "English",
        "draw": "Рисовать",
        "erase": "Ластик",
        "undo": "Отменить",
        "clear": "Очистить",
        "save": "Сохранить U(x)",
        "load": "Загрузить U(x)",
        "preset": "Шаблон",
        "apply": "Применить",
        "preset_flat": "Плоский",
        "preset_well": "Одна яма",
        "preset_double": "Две ямы",
        "preset_barrier": "Энергетический барьер",
        "preset_tilt": "Наклон",
        "blank_hint": "Холст пуст. Нарисуйте U(x) левой кнопкой мыши.",
        "draw_hint": (
            "ЛКМ — выбранный инструмент; ПКМ — временный ластик. "
            "Серый пунктир показывает интерполяцию незаполненных участков."
        ),
        "run": "Старт",
        "pause": "Пауза",
        "reset_particles": "Пересоздать частицы",
        "number": "Число частиц N",
        "radius": "Радиус, пикс.",
        "temperature": "Температура T",
        "friction": "Трение γ",
        "energy_scale": "Масштаб энергии",
        "partition": "Перегородка",
        "partition_none": "Нет",
        "partition_solid": "Сплошная",
        "partition_porous": "Пористая",
        "partition_position": "Положение перегородки",
        "pore_size": "Размер поры, пикс.",
        "shortcuts": "Горячие клавиши",
        "shortcuts_text": (
            "Space — старт/пауза\n"
            "D / E — карандаш / ластик\n"
            "Ctrl+Z — отменить\n"
            "Ctrl+L — очистить\n"
            "Ctrl+S / Ctrl+O — сохранить / открыть\n"
            "R — пересоздать частицы\n"
            "Esc — пауза"
        ),
        "status_running": "моделирование",
        "status_paused": "пауза",
        "center_mass": "центр масс",
        "entropy": "энтропия Sx",
        "time": "время",
        "experiment": "эксперимент",
        "theory": "Больцман",
        "x_axis": "положение x",
        "high_u": "высокая U",
        "low_u": "низкая U",
        "saved": "Потенциал сохранён.",
        "loaded": "Потенциал загружен.",
        "save_error": "Не удалось сохранить файл:\n{error}",
        "load_error": "Не удалось загрузить потенциал:\n{error}",
        "bad_file": "Файл не является потенциалом Boltzmann Lab или повреждён.",
        "about_blank": "Пустой потенциал интерпретируется как постоянный U(x).",
    },

    "en": {
        "app_title": "Boltzmann Lab — Boltzmann distribution",
        "potential_title": "Potential energy U(x): draw with the mouse",
        "particles_title": "Particle motion",
        "distribution_title": "Distribution along x",
        "controls": "Model parameters",
        "language": "Language",
        "russian": "Русский",
        "english": "English",
        "draw": "Draw",
        "erase": "Eraser",
        "undo": "Undo",
        "clear": "Clear",
        "save": "Save U(x)",
        "load": "Load U(x)",
        "preset": "Template",
        "apply": "Apply",
        "preset_flat": "Flat",
        "preset_well": "Single well",
        "preset_double": "Double well",
        "preset_barrier": "Energy barrier",
        "preset_tilt": "Tilt",
        "blank_hint": "Canvas is blank. Draw U(x) with the left mouse button.",
        "draw_hint": (
            "Left button — selected tool; right button — temporary eraser. "
            "A gray dashed line shows interpolation across missing regions."
        ),
        "run": "Start",
        "pause": "Pause",
        "reset_particles": "Reset particles",
        "number": "Number of particles N",
        "radius": "Radius, px",
        "temperature": "Temperature T",
        "friction": "Friction γ",
        "energy_scale": "Energy scale",
        "partition": "Partition",
        "partition_none": "None",
        "partition_solid": "Solid",
        "partition_porous": "Porous",
        "partition_position": "Partition position",
        "pore_size": "Pore size, px",
        "shortcuts": "Keyboard shortcuts",
        "shortcuts_text": (
            "Space — start/pause\n"
            "D / E — draw / erase\n"
            "Ctrl+Z — undo\n"
            "Ctrl+L — clear\n"
            "Ctrl+S / Ctrl+O — save / open\n"
            "R — reset particles\n"
            "Esc — pause"
        ),
        "status_running": "running",
        "status_paused": "paused",
        "center_mass": "center of mass",
        "entropy": "entropy Sx",
        "time": "time",
        "experiment": "experiment",
        "theory": "Boltzmann",
        "x_axis": "position x",
        "high_u": "high U",
        "low_u": "low U",
        "saved": "Potential saved.",
        "loaded": "Potential loaded.",
        "save_error": "Could not save file:\n{error}",
        "load_error": "Could not load potential:\n{error}",
        "bad_file": "The file is not a valid Boltzmann Lab potential.",
        "about_blank": "A blank potential is interpreted as constant U(x).",
    },
}

PRESET_KEYS = ("flat", "well", "double", "barrier", "tilt")
PARTITION_KEYS = ("none", "solid", "porous")


# -----------------------------------------------------------------------------
# Potential model
# -----------------------------------------------------------------------------

class PotentialModel:
    """
    Editable sampled 1D potential.

    values contains normalized U samples in [0, 1].
    None means that the sample has not explicitly been drawn.

    Physics needs a complete U(x), so missing interior regions are linearly
    interpolated, and missing edge regions are extended with the nearest known
    value. If everything is blank, U(x) is treated as constant.
    """

    def __init__(self, points: int = 401, undo_limit: int = 40) -> None:
        self.points = points
        self.values: list[Optional[float]] = [None] * points

        self.undo_limit = undo_limit
        self.undo_stack: list[list[Optional[float]]] = []

        self._compiled_cache: Optional[list[float]] = None
        self._gradient_cache: Optional[list[float]] = None

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def invalidate(self) -> None:
        self._compiled_cache = None
        self._gradient_cache = None

    def is_blank(self) -> bool:
        return not any(v is not None for v in self.values)

    def push_undo(self) -> None:
        self.undo_stack.append(self.values.copy())

        if len(self.undo_stack) > self.undo_limit:
            del self.undo_stack[0]

    def undo(self) -> bool:
        if not self.undo_stack:
            return False

        self.values = self.undo_stack.pop()
        self.invalidate()
        return True

    def clear(self) -> None:
        self.values = [None] * self.points
        self.invalidate()

    def set_draw_segment(
        self,
        i0: int,
        v0: float,
        i1: int,
        v1: float,
    ) -> None:
        """
        Draw a continuous line between two mouse samples.

        Interpolation between mouse events prevents holes when the pointer moves
        several pixels between successive Motion events.
        """
        i0 = max(0, min(self.points - 1, i0))
        i1 = max(0, min(self.points - 1, i1))

        v0 = self._clip01(v0)
        v1 = self._clip01(v1)

        if i0 == i1:
            self.values[i0] = v1
            self.invalidate()
            return

        step = 1 if i1 > i0 else -1
        span = i1 - i0

        for i in range(i0, i1 + step, step):
            alpha = (i - i0) / span
            self.values[i] = self._clip01(
                v0 + alpha * (v1 - v0)
            )

        self.invalidate()

    def erase_segment(
        self,
        i0: int,
        i1: int,
        brush: int = 5,
    ) -> None:
        """Erase sampled potential values with a finite-width brush."""
        lo = max(0, min(i0, i1) - brush)
        hi = min(self.points - 1, max(i0, i1) + brush)

        for i in range(lo, hi + 1):
            self.values[i] = None

        self.invalidate()

    def apply_preset(self, key: str) -> None:
        """Create a predefined potential profile."""
        result: list[float] = []
        n = self.points

        for i in range(n):
            x = i / (n - 1)

            if key == "flat":
                u = 0.5

            elif key == "well":
                u = (
                    0.88
                    - 0.78
                    * math.exp(
                        -((x - 0.5) / 0.14) ** 2
                    )
                )

            elif key == "double":
                well1 = math.exp(
                    -((x - 0.28) / 0.10) ** 2
                )
                well2 = math.exp(
                    -((x - 0.72) / 0.10) ** 2
                )

                u = 0.90 - 0.80 * max(
                    well1,
                    well2,
                )

            elif key == "barrier":
                u = (
                    0.10
                    + 0.82
                    * math.exp(
                        -((x - 0.5) / 0.09) ** 2
                    )
                )

            elif key == "tilt":
                u = 0.08 + 0.84 * x

            else:
                raise ValueError(
                    f"unknown preset: {key}"
                )

            result.append(
                self._clip01(u)
            )

        self.values = result
        self.invalidate()

    def compiled(self) -> list[float]:
        """
        Return the complete potential used by physics.

        Rules:
          * all blank -> constant 0.5;
          * interior holes -> linear interpolation;
          * empty regions at edges -> nearest known value.
        """
        if self._compiled_cache is not None:
            return self._compiled_cache

        known = [
            i
            for i, value in enumerate(self.values)
            if value is not None
        ]

        n = self.points

        if not known:
            out = [0.5] * n
            self._compiled_cache = out
            return out

        out = [0.0] * n

        first = known[0]
        first_value = float(
            self.values[first]
        )

        for i in range(0, first + 1):
            out[i] = first_value

        previous = first

        for current in known[1:]:
            v0 = float(
                self.values[previous]
            )
            v1 = float(
                self.values[current]
            )

            gap = current - previous

            for i in range(
                previous,
                current + 1,
            ):
                alpha = (
                    i - previous
                ) / gap

                out[i] = (
                    v0
                    + alpha
                    * (v1 - v0)
                )

            previous = current

        last = known[-1]
        last_value = float(
            self.values[last]
        )

        for i in range(last, n):
            out[i] = last_value

        self._compiled_cache = out
        return out

    def gradients(self) -> list[float]:
        """
        Finite-difference approximation to du/dx on the sampled grid.
        """
        if self._gradient_cache is not None:
            return self._gradient_cache

        u = self.compiled()

        dx = 1.0 / (
            self.points - 1
        )

        gradient = [
            0.0
        ] * self.points

        gradient[0] = (
            u[1] - u[0]
        ) / dx

        gradient[-1] = (
            u[-1] - u[-2]
        ) / dx

        for i in range(
            1,
            self.points - 1,
        ):
            gradient[i] = (
                u[i + 1]
                - u[i - 1]
            ) / (
                2.0 * dx
            )

        self._gradient_cache = gradient
        return gradient

    def sample(self, x: float) -> float:
        """Linearly interpolate U at arbitrary x in [0, 1]."""
        x = max(
            0.0,
            min(1.0, x),
        )

        data = self.compiled()

        f = x * (
            self.points - 1
        )

        i = int(f)

        if i >= self.points - 1:
            return data[-1]

        alpha = f - i

        return (
            data[i]
            * (1.0 - alpha)
            + data[i + 1]
            * alpha
        )

    def gradient(self, x: float) -> float:
        """Linearly interpolate dU/dx at arbitrary x."""
        x = max(
            0.0,
            min(1.0, x),
        )

        data = self.gradients()

        f = x * (
            self.points - 1
        )

        i = int(f)

        if i >= self.points - 1:
            return data[-1]

        alpha = f - i

        return (
            data[i]
            * (1.0 - alpha)
            + data[i + 1]
            * alpha
        )


# -----------------------------------------------------------------------------
# Particle state
# -----------------------------------------------------------------------------

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------

class BoltzmannLab:
    # Canvas dimensions in pixels.
    BOX_W = 820
    POT_H = 170
    BOX_H = 220
    DIST_H = 185

    HIST_BINS = 48

    # Numerical parameters.
    DT = 0.003
    STEPS_PER_FRAME = 4

    # Freehand curves can contain nearly vertical slopes.
    # This clamp protects the explicit force step from numerical explosions.
    FORCE_LIMIT = 120.0

    def __init__(
        self,
        root: tk.Tk,
    ) -> None:
        self.root = root

        self.lang = "ru"

        self.running = False
        self.sim_time = 0.0
        self.frame_no = 0

        # World x width is 1.0.
        # y uses the same pixel-to-world scale, preserving circular particles.
        self.world_height = (
            self.BOX_H
            / self.BOX_W
        )

        self.potential = PotentialModel(
            points=401
        )

        self.particles: list[
            Particle
        ] = []

        self.particle_items: list[
            int
        ] = []

        self.center_line_item: Optional[
            int
        ] = None

        # Exponentially averaged experimental histogram.
        self.hist_ema: Optional[
            list[float]
        ] = None

        # Mouse drawing state.
        self.stroke_active = False
        self.stroke_tool = "draw"

        self.stroke_last: Optional[
            tuple[int, float]
        ] = None

        # Tk variables.
        self.tool_var = tk.StringVar(
            value="draw"
        )

        self.lang_var = tk.StringVar(
            value="Русский"
        )

        self.preset_var = tk.StringVar(
            value=""
        )

        self.partition_var = tk.StringVar(
            value=""
        )

        self.n_var = tk.IntVar(
            value=140
        )

        self.radius_var = tk.DoubleVar(
            value=4.0
        )

        self.temp_var = tk.DoubleVar(
            value=1.0
        )

        self.gamma_var = tk.DoubleVar(
            value=2.0
        )

        self.energy_var = tk.DoubleVar(
            value=4.0
        )

        self.partition_x_var = (
            tk.DoubleVar(
                value=0.50
            )
        )

        self.pore_var = tk.DoubleVar(
            value=34.0
        )

        self.partition_mode = "none"

        self.labels: dict[
            str,
            ttk.Label,
        ] = {}

        self.buttons: dict[
            str,
            ttk.Button,
        ] = {}

        self.scales: dict[
            str,
            tk.Scale,
        ] = {}

        self._setup_window()
        self._build_ui()
        self._bind_shortcuts()

        self._update_language()

        self.reset_particles()

        self._redraw_potential()
        self._redraw_partition_and_particles()

        self._redraw_distribution(
            update_ema=False
        )

        self._update_stats()

        # Schedule the first animation callback.
        self.root.after(
            16,
            self._animation_loop,
        )

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def tr(
        self,
        key: str,
    ) -> str:
        return TEXT[
            self.lang
        ][key]

    def _setup_window(
        self,
    ) -> None:
        self.root.geometry(
            "1180x800"
        )

        self.root.minsize(
            1080,
            740,
        )

        try:
            ttk.Style().theme_use(
                "clam"
            )
        except tk.TclError:
            pass

        style = ttk.Style()

        style.configure(
            "Title.TLabel",
            font=(
                "TkDefaultFont",
                11,
                "bold",
            ),
        )

        style.configure(
            "Hint.TLabel",
            foreground="#555555",
        )

        style.configure(
            "Stats.TLabel",
            font=(
                "TkFixedFont",
                10,
            ),
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(
        self,
    ) -> None:
        outer = ttk.Frame(
            self.root,
            padding=8,
        )

        outer.pack(
            fill="both",
            expand=True,
        )

        left = ttk.Frame(
            outer
        )

        left.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.control_frame = (
            ttk.LabelFrame(
                outer,
                padding=10,
            )
        )

        self.control_frame.pack(
            side="right",
            fill="y",
            padx=(10, 0),
        )

        # --------------------------------------------------------------
        # Potential title
        # --------------------------------------------------------------

        self.labels[
            "potential_title"
        ] = ttk.Label(
            left,
            style="Title.TLabel",
        )

        self.labels[
            "potential_title"
        ].pack(
            anchor="w"
        )

        # --------------------------------------------------------------
        # Potential toolbar
        # --------------------------------------------------------------

        toolbar = ttk.Frame(
            left
        )

        toolbar.pack(
            fill="x",
            pady=(3, 3),
        )

        self.draw_radio = (
            ttk.Radiobutton(
                toolbar,
                variable=self.tool_var,
                value="draw",
            )
        )

        self.draw_radio.grid(
            row=0,
            column=0,
            padx=(0, 4),
        )

        self.erase_radio = (
            ttk.Radiobutton(
                toolbar,
                variable=self.tool_var,
                value="erase",
            )
        )

        self.erase_radio.grid(
            row=0,
            column=1,
            padx=4,
        )

        self.buttons[
            "undo"
        ] = ttk.Button(
            toolbar,
            command=self.undo_potential,
        )

        self.buttons[
            "undo"
        ].grid(
            row=0,
            column=2,
            padx=4,
        )

        self.buttons[
            "clear"
        ] = ttk.Button(
            toolbar,
            command=self.clear_potential,
        )

        self.buttons[
            "clear"
        ].grid(
            row=0,
            column=3,
            padx=4,
        )

        self.labels[
            "preset"
        ] = ttk.Label(
            toolbar
        )

        self.labels[
            "preset"
        ].grid(
            row=0,
            column=4,
            padx=(14, 4),
        )

        self.preset_combo = (
            ttk.Combobox(
                toolbar,
                textvariable=self.preset_var,
                state="readonly",
                width=18,
            )
        )

        self.preset_combo.grid(
            row=0,
            column=5,
            padx=4,
        )

        self.buttons[
            "apply"
        ] = ttk.Button(
            toolbar,
            command=self.apply_selected_preset,
        )

        self.buttons[
            "apply"
        ].grid(
            row=0,
            column=6,
            padx=4,
        )

        self.buttons[
            "save"
        ] = ttk.Button(
            toolbar,
            command=self.save_potential,
        )

        self.buttons[
            "save"
        ].grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(4, 0),
            padx=(0, 4),
        )

        self.buttons[
            "load"
        ] = ttk.Button(
            toolbar,
            command=self.load_potential,
        )

        self.buttons[
            "load"
        ].grid(
            row=1,
            column=2,
            columnspan=2,
            sticky="ew",
            pady=(4, 0),
            padx=4,
        )

        # --------------------------------------------------------------
        # Potential canvas
        # --------------------------------------------------------------

        self.potential_canvas = (
            tk.Canvas(
                left,
                width=self.BOX_W,
                height=self.POT_H,
                background="white",
                highlightthickness=1,
                highlightbackground="#777777",
                cursor="crosshair",
            )
        )

        self.potential_canvas.pack(
            pady=(2, 2)
        )

        self.potential_canvas.bind(
            "<Button-1>",
            self._begin_draw,
        )

        self.potential_canvas.bind(
            "<B1-Motion>",
            self._continue_draw,
        )

        self.potential_canvas.bind(
            "<ButtonRelease-1>",
            self._end_draw,
        )

        # Right button always works as a temporary eraser.
        self.potential_canvas.bind(
            "<Button-3>",
            self._begin_right_erase,
        )

        self.potential_canvas.bind(
            "<B3-Motion>",
            self._continue_draw,
        )

        self.potential_canvas.bind(
            "<ButtonRelease-3>",
            self._end_draw,
        )

        self.labels[
            "draw_hint"
        ] = ttk.Label(
            left,
            style="Hint.TLabel",
            wraplength=self.BOX_W,
        )

        self.labels[
            "draw_hint"
        ].pack(
            anchor="w",
            pady=(0, 4),
        )

        # --------------------------------------------------------------
        # Particle canvas
        # --------------------------------------------------------------

        self.labels[
            "particles_title"
        ] = ttk.Label(
            left,
            style="Title.TLabel",
        )

        self.labels[
            "particles_title"
        ].pack(
            anchor="w"
        )

        self.particle_canvas = (
            tk.Canvas(
                left,
                width=self.BOX_W,
                height=self.BOX_H,
                background="#f7fbff",
                highlightthickness=1,
                highlightbackground="#555555",
            )
        )

        self.particle_canvas.pack(
            pady=(2, 3)
        )

        # --------------------------------------------------------------
        # Statistics label
        # --------------------------------------------------------------

        self.stats_var = (
            tk.StringVar()
        )

        ttk.Label(
            left,
            textvariable=self.stats_var,
            style="Stats.TLabel",
        ).pack(
            anchor="w",
            pady=(0, 4),
        )

        # --------------------------------------------------------------
        # Distribution canvas
        # --------------------------------------------------------------

        self.labels[
            "distribution_title"
        ] = ttk.Label(
            left,
            style="Title.TLabel",
        )

        self.labels[
            "distribution_title"
        ].pack(
            anchor="w"
        )

        self.distribution_canvas = (
            tk.Canvas(
                left,
                width=self.BOX_W,
                height=self.DIST_H,
                background="white",
                highlightthickness=1,
                highlightbackground="#777777",
            )
        )

        self.distribution_canvas.pack(
            pady=(2, 0)
        )

        # --------------------------------------------------------------
        # Right-side control panel
        # --------------------------------------------------------------

        row = 0

        self.labels[
            "language"
        ] = ttk.Label(
            self.control_frame
        )

        self.labels[
            "language"
        ].grid(
            row=row,
            column=0,
            sticky="w",
        )

        row += 1

        self.lang_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.lang_var,
            values=(
                "Русский",
                "English",
            ),
            state="readonly",
            width=24,
        )

        self.lang_combo.grid(
            row=row,
            column=0,
            sticky="ew",
            pady=(2, 8),
        )

        self.lang_combo.bind(
            "<<ComboboxSelected>>",
            self._language_changed,
        )

        row += 1

        self.buttons[
            "run"
        ] = ttk.Button(
            self.control_frame,
            command=self.toggle_running,
        )

        self.buttons[
            "run"
        ].grid(
            row=row,
            column=0,
            sticky="ew",
            pady=2,
        )

        row += 1

        self.buttons[
            "reset_particles"
        ] = ttk.Button(
            self.control_frame,
            command=self.reset_particles,
        )

        self.buttons[
            "reset_particles"
        ].grid(
            row=row,
            column=0,
            sticky="ew",
            pady=(2, 8),
        )

        row += 1

        self.scales[
            "number"
        ] = self._make_scale(
            row,
            self.n_var,
            20,
            300,
            1,
        )

        self.scales[
            "number"
        ].bind(
            "<ButtonRelease-1>",
            lambda _event:
                self.reset_particles(),
        )

        row += 1

        self.scales[
            "radius"
        ] = self._make_scale(
            row,
            self.radius_var,
            2,
            8,
            1,
        )

        self.scales[
            "radius"
        ].bind(
            "<ButtonRelease-1>",
            lambda _event:
                self.reset_particles(),
        )

        row += 1

        self.scales[
            "temperature"
        ] = self._make_scale(
            row,
            self.temp_var,
            0.10,
            3.00,
            0.05,
        )

        self.scales[
            "temperature"
        ].bind(
            "<ButtonRelease-1>",
            lambda _event:
                self._equilibrium_parameter_changed(),
        )

        row += 1

        self.scales[
            "friction"
        ] = self._make_scale(
            row,
            self.gamma_var,
            0.10,
            8.00,
            0.10,
        )

        row += 1

        self.scales[
            "energy_scale"
        ] = self._make_scale(
            row,
            self.energy_var,
            0.0,
            12.0,
            0.10,
        )

        self.scales[
            "energy_scale"
        ].bind(
            "<ButtonRelease-1>",
            lambda _event:
                self._equilibrium_parameter_changed(),
        )

        row += 1

        ttk.Separator(
            self.control_frame,
            orient="horizontal",
        ).grid(
            row=row,
            column=0,
            sticky="ew",
            pady=7,
        )

        row += 1

        self.labels[
            "partition"
        ] = ttk.Label(
            self.control_frame
        )

        self.labels[
            "partition"
        ].grid(
            row=row,
            column=0,
            sticky="w",
        )

        row += 1

        self.partition_combo = (
            ttk.Combobox(
                self.control_frame,
                textvariable=self.partition_var,
                state="readonly",
                width=24,
            )
        )

        self.partition_combo.grid(
            row=row,
            column=0,
            sticky="ew",
            pady=(2, 6),
        )

        self.partition_combo.bind(
            "<<ComboboxSelected>>",
            self._partition_changed,
        )

        row += 1

        self.scales[
            "partition_position"
        ] = self._make_scale(
            row,
            self.partition_x_var,
            0.10,
            0.90,
            0.01,
            command=lambda _value:
                self._partition_geometry_changed(
                    live=True
                ),
        )

        self.scales[
            "partition_position"
        ].bind(
            "<ButtonRelease-1>",
            lambda _event:
                self._partition_geometry_changed(
                    live=False
                ),
        )

        row += 1

        self.scales[
            "pore_size"
        ] = self._make_scale(
            row,
            self.pore_var,
            4,
            90,
            1,
            command=lambda _value:
                self._partition_geometry_changed(
                    live=True
                ),
        )

        self.scales[
            "pore_size"
        ].bind(
            "<ButtonRelease-1>",
            lambda _event:
                self._partition_geometry_changed(
                    live=False
                ),
        )

        row += 1

        ttk.Separator(
            self.control_frame,
            orient="horizontal",
        ).grid(
            row=row,
            column=0,
            sticky="ew",
            pady=7,
        )

        row += 1

        self.labels[
            "shortcuts"
        ] = ttk.Label(
            self.control_frame,
            style="Title.TLabel",
        )

        self.labels[
            "shortcuts"
        ].grid(
            row=row,
            column=0,
            sticky="w",
        )

        row += 1

        self.labels[
            "shortcuts_text"
        ] = ttk.Label(
            self.control_frame,
            justify="left",
            style="Hint.TLabel",
            wraplength=255,
        )

        self.labels[
            "shortcuts_text"
        ].grid(
            row=row,
            column=0,
            sticky="w",
        )

        self.control_frame.columnconfigure(
            0,
            weight=1,
        )

    def _make_scale(
        self,
        row: int,
        variable: tk.Variable,
        minimum: float,
        maximum: float,
        resolution: float,
        command=None,
    ) -> tk.Scale:
        scale = tk.Scale(
            self.control_frame,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient="horizontal",
            length=260,
            showvalue=True,
            highlightthickness=0,
            command=command,
        )

        scale.grid(
            row=row,
            column=0,
            sticky="ew",
            pady=1,
        )

        return scale

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _bind_shortcuts(
        self,
    ) -> None:
        self.root.bind_all(
            "<space>",
            lambda _event:
                self._shortcut(
                    self.toggle_running
                ),
        )

        self.root.bind_all(
            "<Control-z>",
            lambda _event:
                self._shortcut(
                    self.undo_potential
                ),
        )

        self.root.bind_all(
            "<Control-s>",
            lambda _event:
                self._shortcut(
                    self.save_potential
                ),
        )

        self.root.bind_all(
            "<Control-o>",
            lambda _event:
                self._shortcut(
                    self.load_potential
                ),
        )

        self.root.bind_all(
            "<Control-l>",
            lambda _event:
                self._shortcut(
                    self.clear_potential
                ),
        )

        self.root.bind_all(
            "<KeyPress-d>",
            lambda _event:
                self._shortcut(
                    lambda:
                        self.tool_var.set(
                            "draw"
                        )
                ),
        )

        self.root.bind_all(
            "<KeyPress-e>",
            lambda _event:
                self._shortcut(
                    lambda:
                        self.tool_var.set(
                            "erase"
                        )
                ),
        )

        self.root.bind_all(
            "<KeyPress-r>",
            lambda _event:
                self._shortcut(
                    self.reset_particles
                ),
        )

        self.root.bind_all(
            "<Escape>",
            lambda _event:
                self._shortcut(
                    self.pause
                ),
        )

    @staticmethod
    def _shortcut(
        action,
    ) -> str:
        action()
        return "break"

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def _language_changed(
        self,
        _event=None,
    ) -> None:
        self.lang = (
            "ru"
            if self.lang_var.get()
            == "Русский"
            else "en"
        )

        self._update_language()

    def _update_language(
        self,
    ) -> None:
        self.root.title(
            self.tr(
                "app_title"
            )
        )

        self.control_frame.configure(
            text=self.tr(
                "controls"
            )
        )

        for key in (
            "potential_title",
            "particles_title",
            "distribution_title",
            "preset",
            "language",
            "partition",
            "shortcuts",
            "shortcuts_text",
        ):
            self.labels[key].configure(
                text=self.tr(key)
            )

        self.labels[
            "draw_hint"
        ].configure(
            text=(
                self.tr("draw_hint")
                + "  "
                + self.tr("about_blank")
            )
        )

        self.draw_radio.configure(
            text=self.tr(
                "draw"
            )
        )

        self.erase_radio.configure(
            text=self.tr(
                "erase"
            )
        )

        for key in (
            "undo",
            "clear",
            "save",
            "load",
            "apply",
            "reset_particles",
        ):
            self.buttons[
                key
            ].configure(
                text=self.tr(key)
            )

        self._update_run_button()

        self.scales[
            "number"
        ].configure(
            label=self.tr(
                "number"
            )
        )

        self.scales[
            "radius"
        ].configure(
            label=self.tr(
                "radius"
            )
        )

        self.scales[
            "temperature"
        ].configure(
            label=self.tr(
                "temperature"
            )
        )

        self.scales[
            "friction"
        ].configure(
            label=self.tr(
                "friction"
            )
        )

        self.scales[
            "energy_scale"
        ].configure(
            label=self.tr(
                "energy_scale"
            )
        )

        self.scales[
            "partition_position"
        ].configure(
            label=self.tr(
                "partition_position"
            )
        )

        self.scales[
            "pore_size"
        ].configure(
            label=self.tr(
                "pore_size"
            )
        )

        preset_labels = [
            self.tr(
                f"preset_{key}"
            )
            for key in PRESET_KEYS
        ]

        old_preset = (
            self._selected_preset_key()
        )

        self.preset_combo.configure(
            values=preset_labels
        )

        if old_preset:
            self.preset_var.set(
                self.tr(
                    f"preset_{old_preset}"
                )
            )

        elif not self.preset_var.get():
            self.preset_var.set(
                self.tr(
                    "preset_double"
                )
            )

        partition_labels = [
            self.tr(
                f"partition_{key}"
            )
            for key in PARTITION_KEYS
        ]

        self.partition_combo.configure(
            values=partition_labels
        )

        self.partition_var.set(
            self.tr(
                f"partition_{self.partition_mode}"
            )
        )

        self._redraw_potential()
        self._redraw_partition_and_particles()

        self._redraw_distribution(
            update_ema=False
        )

        self._update_stats()

    def _selected_preset_key(
        self,
    ) -> Optional[str]:
        text = (
            self.preset_var.get()
        )

        for lang in (
            "ru",
            "en",
        ):
            for key in PRESET_KEYS:
                if (
                    text
                    == TEXT[
                        lang
                    ][
                        f"preset_{key}"
                    ]
                ):
                    return key

        return None

    # ------------------------------------------------------------------
    # Potential editing
    # ------------------------------------------------------------------

    def _event_to_potential(
        self,
        event: tk.Event,
    ) -> tuple[int, float]:
        top = 12
        bottom = (
            self.POT_H - 24
        )

        x = max(
            0,
            min(
                self.BOX_W - 1,
                int(event.x),
            ),
        )

        y = max(
            top,
            min(
                bottom,
                int(event.y),
            ),
        )

        index = round(
            x
            / (
                self.BOX_W - 1
            )
            * (
                self.potential.points
                - 1
            )
        )

        value = (
            bottom - y
        ) / (
            bottom - top
        )

        return (
            index,
            max(
                0.0,
                min(
                    1.0,
                    value,
                ),
            ),
        )

    def _begin_draw(
        self,
        event: tk.Event,
    ) -> None:
        self._start_stroke(
            event,
            self.tool_var.get(),
        )

    def _begin_right_erase(
        self,
        event: tk.Event,
    ) -> None:
        self._start_stroke(
            event,
            "erase",
        )

    def _start_stroke(
        self,
        event: tk.Event,
        tool: str,
    ) -> None:
        # One complete mouse stroke corresponds to one undo state.
        self.potential.push_undo()

        self.stroke_active = True
        self.stroke_tool = tool

        index, value = (
            self._event_to_potential(
                event
            )
        )

        self.stroke_last = (
            index,
            value,
        )

        if tool == "erase":
            self.potential.erase_segment(
                index,
                index,
            )

        else:
            self.potential.set_draw_segment(
                index,
                value,
                index,
                value,
            )

        self._potential_changed()

    def _continue_draw(
        self,
        event: tk.Event,
    ) -> None:
        if (
            not self.stroke_active
            or self.stroke_last
            is None
        ):
            return

        index, value = (
            self._event_to_potential(
                event
            )
        )

        old_index, old_value = (
            self.stroke_last
        )

        if (
            self.stroke_tool
            == "erase"
        ):
            self.potential.erase_segment(
                old_index,
                index,
            )

        else:
            self.potential.set_draw_segment(
                old_index,
                old_value,
                index,
                value,
            )

        self.stroke_last = (
            index,
            value,
        )

        self._potential_changed()

    def _end_draw(
        self,
        _event=None,
    ) -> None:
        self.stroke_active = False
        self.stroke_last = None

    def _potential_changed(
        self,
    ) -> None:
        # Previous histogram samples belong to the previous external field.
        self.hist_ema = None

        self._redraw_potential()

        self._redraw_distribution(
            update_ema=False
        )

    def undo_potential(
        self,
    ) -> None:
        if self.potential.undo():
            self._potential_changed()

    def clear_potential(
        self,
    ) -> None:
        self.potential.push_undo()
        self.potential.clear()
        self._potential_changed()

    def apply_selected_preset(
        self,
    ) -> None:
        key = (
            self._selected_preset_key()
            or "double"
        )

        self.potential.push_undo()

        self.potential.apply_preset(
            key
        )

        self._potential_changed()

    # ------------------------------------------------------------------
    # Save / load potential
    # ------------------------------------------------------------------

    def save_potential(
        self,
    ) -> None:
        filename = (
            filedialog.asksaveasfilename(
                title=self.tr(
                    "save"
                ),
                defaultextension=".json",
                filetypes=(
                    (
                        "JSON",
                        "*.json",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ),
            )
        )

        if not filename:
            return

        payload = {
            "format":
                "BoltzmannLabPotential",
            "version": 1,
            "grid_points":
                self.potential.points,
            "values":
                self.potential.values,
        }

        try:
            with open(
                filename,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

        except (
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            messagebox.showerror(
                self.tr("save"),
                self.tr(
                    "save_error"
                ).format(
                    error=exc
                ),
            )

            return

        self.root.bell()

    def load_potential(
        self,
    ) -> None:
        filename = (
            filedialog.askopenfilename(
                title=self.tr(
                    "load"
                ),
                filetypes=(
                    (
                        "JSON",
                        "*.json",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ),
            )
        )

        if not filename:
            return

        try:
            with open(
                filename,
                "r",
                encoding="utf-8",
            ) as handle:
                payload = (
                    json.load(
                        handle
                    )
                )

            if (
                not isinstance(
                    payload,
                    dict,
                )
                or payload.get(
                    "format"
                )
                != "BoltzmannLabPotential"
                or payload.get(
                    "version"
                )
                != 1
            ):
                raise ValueError(
                    self.tr(
                        "bad_file"
                    )
                )

            values = payload.get(
                "values"
            )

            if (
                not isinstance(
                    values,
                    list,
                )
                or len(values)
                != self.potential.points
            ):
                raise ValueError(
                    self.tr(
                        "bad_file"
                    )
                )

            checked: list[
                Optional[float]
            ] = []

            for value in values:
                if value is None:
                    checked.append(
                        None
                    )

                elif (
                    isinstance(
                        value,
                        (
                            int,
                            float,
                        ),
                    )
                    and math.isfinite(
                        float(value)
                    )
                ):
                    checked.append(
                        max(
                            0.0,
                            min(
                                1.0,
                                float(
                                    value
                                ),
                            ),
                        )
                    )

                else:
                    raise ValueError(
                        self.tr(
                            "bad_file"
                        )
                    )

        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
        ) as exc:
            messagebox.showerror(
                self.tr("load"),
                self.tr(
                    "load_error"
                ).format(
                    error=exc
                ),
            )

            return

        self.potential.push_undo()

        self.potential.values = (
            checked
        )

        self.potential.invalidate()

        self._potential_changed()

    # ------------------------------------------------------------------
    # Simulation controls
    # ------------------------------------------------------------------

    def toggle_running(
        self,
    ) -> None:
        self.running = (
            not self.running
        )

        self._update_run_button()
        self._update_stats()

    def pause(
        self,
    ) -> None:
        self.running = False
        self._update_run_button()
        self._update_stats()

    def _update_run_button(
        self,
    ) -> None:
        self.buttons[
            "run"
        ].configure(
            text=(
                self.tr("pause")
                if self.running
                else self.tr("run")
            )
        )

    def _equilibrium_parameter_changed(
        self,
    ) -> None:
        self.hist_ema = None

        self._redraw_distribution(
            update_ema=False
        )

    # ------------------------------------------------------------------
    # Partitions and pores
    # ------------------------------------------------------------------

    def _partition_changed(
        self,
        _event=None,
    ) -> None:
        selected = (
            self.partition_var.get()
        )

        for lang in (
            "ru",
            "en",
        ):
            for key in PARTITION_KEYS:
                if (
                    selected
                    == TEXT[
                        lang
                    ][
                        f"partition_{key}"
                    ]
                ):
                    self.partition_mode = (
                        key
                    )
                    break

        self.hist_ema = None

        self._enforce_all_constraints()
        self._redraw_partition_and_particles()

        self._redraw_distribution(
            update_ema=False
        )

    def _partition_geometry_changed(
        self,
        live: bool,
    ) -> None:
        if not live:
            self._enforce_all_constraints()
            self.hist_ema = None

        self._redraw_partition_and_particles()

    def _pore_allows_y(
        self,
        y_world: float,
        radius_px: Optional[
            float
        ] = None,
    ) -> bool:
        """
        Return True if a circular particle fits through the pore at this y.

        Openings and solid pieces alternate vertically with equal nominal height.
        The center of a particle must remain at least one radius away from both
        edges of an opening. This naturally implements the diameter filter.
        """
        if radius_px is None:
            radius_px = (
                self.radius_var.get()
            )

        pore = float(
            self.pore_var.get()
        )

        # The complete diameter must fit.
        if (
            pore
            <= 2.0 * radius_px
        ):
            return False

        y_px = (
            y_world
            * self.BOX_W
        )

        period = (
            2.0 * pore
        )

        phase = (
            y_px % period
        )

        return (
            radius_px
            <= phase
            <= pore - radius_px
        )

    def _partition_blocks_y(
        self,
        y_world: float,
        radius_px: Optional[
            float
        ] = None,
    ) -> bool:
        if (
            self.partition_mode
            == "none"
        ):
            return False

        if (
            self.partition_mode
            == "solid"
        ):
            return True

        return not self._pore_allows_y(
            y_world,
            radius_px,
        )

    def _enforce_all_constraints(
        self,
    ) -> None:
        for particle in self.particles:
            self._enforce_constraints(
                particle
            )

    def _enforce_constraints(
        self,
        particle: Particle,
    ) -> None:
        """
        Clamp a particle inside the outer box and push it outside solid pieces
        of the vertical partition if necessary.
        """
        radius = (
            self.particle_radius_world()
        )

        particle.x = max(
            radius,
            min(
                1.0 - radius,
                particle.x,
            ),
        )

        particle.y = max(
            radius,
            min(
                self.world_height
                - radius,
                particle.y,
            ),
        )

        if (
            self.partition_mode
            == "none"
        ):
            return

        bx = float(
            self.partition_x_var.get()
        )

        if (
            abs(
                particle.x - bx
            )
            < radius
            and self._partition_blocks_y(
                particle.y
            )
        ):
            if particle.x < bx:
                particle.x = (
                    bx - radius
                )

                particle.vx = (
                    -abs(
                        particle.vx
                    )
                )

            else:
                particle.x = (
                    bx + radius
                )

                particle.vx = abs(
                    particle.vx
                )

            particle.x = max(
                radius,
                min(
                    1.0 - radius,
                    particle.x,
                ),
            )

    # ------------------------------------------------------------------
    # Particle initialization
    # ------------------------------------------------------------------

    def particle_radius_world(
        self,
    ) -> float:
        return (
            float(
                self.radius_var.get()
            )
            / self.BOX_W
        )

    def reset_particles(
        self,
    ) -> None:
        self.particles.clear()

        self.sim_time = 0.0
        self.hist_ema = None

        count = int(
            self.n_var.get()
        )

        radius = (
            self.particle_radius_world()
        )

        sigma = math.sqrt(
            max(
                0.001,
                float(
                    self.temp_var.get()
                ),
            )
        )

        for _ in range(count):
            x = 0.0
            y = 0.0

            # Try to place particles without initial overlap.
            for _attempt in range(80):
                x = random.uniform(
                    radius,
                    1.0 - radius,
                )

                y = random.uniform(
                    radius,
                    self.world_height
                    - radius,
                )

                if (
                    self.partition_mode
                    != "none"
                    and abs(
                        x
                        - self.partition_x_var.get()
                    )
                    < radius
                    and self._partition_blocks_y(
                        y
                    )
                ):
                    continue

                good = True

                # Initialization is infrequent, so a direct overlap test is
                # acceptable here. The time-critical path uses spatial hashing.
                for other in self.particles:
                    dx = (
                        x - other.x
                    )

                    dy = (
                        y - other.y
                    )

                    if (
                        dx * dx
                        + dy * dy
                        < (
                            2.02
                            * radius
                        ) ** 2
                    ):
                        good = False
                        break

                if good:
                    break

            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=random.gauss(
                        0.0,
                        sigma,
                    ),
                    vy=random.gauss(
                        0.0,
                        sigma,
                    ),
                )
            )

        # Remove accidental fallback overlaps.
        for _ in range(5):
            self._resolve_collisions()

        self._redraw_partition_and_particles()

        self._redraw_distribution(
            update_ema=False
        )

        self._update_stats()

    # ------------------------------------------------------------------
    # Force from the user-defined potential
    # ------------------------------------------------------------------

    def _force_x(
        self,
        x: float,
    ) -> float:
        force = (
            -float(
                self.energy_var.get()
            )
            * self.potential.gradient(
                x
            )
        )

        # Hand-drawn almost-vertical segments can imply a huge finite-difference
        # derivative. Clipping is only a numerical guard for such pathological
        # input; smooth curves normally remain far below the limit.
        return max(
            -self.FORCE_LIMIT,
            min(
                self.FORCE_LIMIT,
                force,
            ),
        )

    # ------------------------------------------------------------------
    # Particle drift and reflections
    # ------------------------------------------------------------------

    def _drift(
        self,
        particle: Particle,
        dt: float,
    ) -> None:
        radius = (
            self.particle_radius_world()
        )

        old_x = particle.x

        particle.x += (
            particle.vx * dt
        )

        particle.y += (
            particle.vy * dt
        )

        # --------------------------------------------------------------
        # Outer walls
        # --------------------------------------------------------------

        if particle.x < radius:
            particle.x = (
                radius
                + (
                    radius
                    - particle.x
                )
            )

            particle.vx = abs(
                particle.vx
            )

        elif (
            particle.x
            > 1.0 - radius
        ):
            particle.x = (
                1.0 - radius
            ) - (
                particle.x
                - (
                    1.0 - radius
                )
            )

            particle.vx = (
                -abs(
                    particle.vx
                )
            )

        if particle.y < radius:
            particle.y = (
                radius
                + (
                    radius
                    - particle.y
                )
            )

            particle.vy = abs(
                particle.vy
            )

        elif (
            particle.y
            > self.world_height
            - radius
        ):
            particle.y = (
                self.world_height
                - radius
            ) - (
                particle.y
                - (
                    self.world_height
                    - radius
                )
            )

            particle.vy = (
                -abs(
                    particle.vy
                )
            )

        # --------------------------------------------------------------
        # Internal partition
        # --------------------------------------------------------------

        if (
            self.partition_mode
            != "none"
            and self._partition_blocks_y(
                particle.y
            )
        ):
            bx = float(
                self.partition_x_var.get()
            )

            left_contact = (
                bx - radius
            )

            right_contact = (
                bx + radius
            )

            # Left -> right collision with a solid section.
            if (
                old_x
                <= left_contact
                < particle.x
                and particle.vx > 0.0
            ):
                overshoot = (
                    particle.x
                    - left_contact
                )

                particle.x = (
                    left_contact
                    - overshoot
                )

                particle.vx = (
                    -abs(
                        particle.vx
                    )
                )

            # Right -> left collision with a solid section.
            elif (
                old_x
                >= right_contact
                > particle.x
                and particle.vx < 0.0
            ):
                overshoot = (
                    right_contact
                    - particle.x
                )

                particle.x = (
                    right_contact
                    + overshoot
                )

                particle.vx = abs(
                    particle.vx
                )

        self._enforce_constraints(
            particle
        )

    # ------------------------------------------------------------------
    # BAOAB-like Langevin integrator
    # ------------------------------------------------------------------

    def _simulation_step(
        self,
    ) -> None:
        """
        One BAOAB-like Langevin step, using m = k_B = 1.

        B: half deterministic force impulse
        A: half coordinate drift
        O: exact Ornstein-Uhlenbeck velocity thermostat
        A: second half coordinate drift
        C: hard-disk collision correction
        B: second half deterministic force impulse

        Hard collisions are an additional operation, so this is deliberately
        described as BAOAB-like rather than an exact textbook BAOAB system.
        """
        dt = self.DT
        half = 0.5 * dt

        # --------------------------------------------------------------
        # B: half force kick
        # --------------------------------------------------------------

        for particle in self.particles:
            particle.vx += (
                half
                * self._force_x(
                    particle.x
                )
            )

        # --------------------------------------------------------------
        # A: half drift
        # --------------------------------------------------------------

        for particle in self.particles:
            self._drift(
                particle,
                half,
            )

        # --------------------------------------------------------------
        # O: exact Ornstein-Uhlenbeck thermostat
        #
        # dv = -gamma*v*dt + sqrt(2*gamma*T)*dW
        #
        # Exact finite-time update:
        #   v_new = c*v_old + sqrt(T*(1-c^2))*R
        #   c = exp(-gamma*dt)
        #   R ~ N(0,1)
        # --------------------------------------------------------------

        gamma = max(
            1e-6,
            float(
                self.gamma_var.get()
            ),
        )

        temperature = max(
            1e-6,
            float(
                self.temp_var.get()
            ),
        )

        c = math.exp(
            -gamma * dt
        )

        noise_sigma = math.sqrt(
            temperature
            * max(
                0.0,
                1.0 - c * c,
            )
        )

        for particle in self.particles:
            particle.vx = (
                c * particle.vx
                + noise_sigma
                * random.gauss(
                    0.0,
                    1.0,
                )
            )

            particle.vy = (
                c * particle.vy
                + noise_sigma
                * random.gauss(
                    0.0,
                    1.0,
                )
            )

        # --------------------------------------------------------------
        # A: second half drift
        # --------------------------------------------------------------

        for particle in self.particles:
            self._drift(
                particle,
                half,
            )

        # Hard-disk collisions.
        self._resolve_collisions()

        # --------------------------------------------------------------
        # B: second half force kick
        # --------------------------------------------------------------

        for particle in self.particles:
            particle.vx += (
                half
                * self._force_x(
                    particle.x
                )
            )

        self.sim_time += dt

    # ------------------------------------------------------------------
    # Elastic particle collisions
    # ------------------------------------------------------------------

    def _partition_separates_pair(
        self,
        particle1: Particle,
        particle2: Particle,
    ) -> bool:
        """
        Prevent particles from colliding "through" a solid partition section.
        """
        if (
            self.partition_mode
            == "none"
        ):
            return False

        bx = float(
            self.partition_x_var.get()
        )

        if (
            (
                particle1.x - bx
            )
            * (
                particle2.x - bx
            )
            >= 0.0
        ):
            return False

        y_mid = 0.5 * (
            particle1.y
            + particle2.y
        )

        return (
            self._partition_blocks_y(
                y_mid
            )
        )

    def _resolve_collisions(
        self,
    ) -> None:
        """
        Resolve equal-mass elastic hard-disk collisions using a spatial hash.

        Instead of examining every possible pair, particles are placed in
        square grid cells. Since the cell width is at least one diameter, a
        particle can overlap only particles in its own or eight neighboring
        cells.

        Collision response:
          * remove overlap symmetrically along contact normal;
          * if particles approach each other, apply the elastic impulse for
            equal masses and restitution coefficient e = 1.
        """
        count = len(
            self.particles
        )

        if count < 2:
            return

        radius = (
            self.particle_radius_world()
        )

        diameter = (
            2.0 * radius
        )

        cell = max(
            diameter * 1.05,
            0.008,
        )

        grid: dict[
            tuple[int, int],
            list[int],
        ] = {}

        for index, particle in enumerate(
            self.particles
        ):
            key = (
                int(
                    particle.x
                    / cell
                ),
                int(
                    particle.y
                    / cell
                ),
            )

            grid.setdefault(
                key,
                [],
            ).append(
                index
            )

        limit2 = (
            diameter
            * diameter
        )

        for i, particle1 in enumerate(
            self.particles
        ):
            gx = int(
                particle1.x
                / cell
            )

            gy = int(
                particle1.y
                / cell
            )

            for offset_x in (
                -1,
                0,
                1,
            ):
                for offset_y in (
                    -1,
                    0,
                    1,
                ):
                    neighbor_indices = (
                        grid.get(
                            (
                                gx
                                + offset_x,
                                gy
                                + offset_y,
                            ),
                            (),
                        )
                    )

                    for j in neighbor_indices:
                        if j <= i:
                            continue

                        particle2 = (
                            self.particles[
                                j
                            ]
                        )

                        if (
                            self._partition_separates_pair(
                                particle1,
                                particle2,
                            )
                        ):
                            continue

                        dx = (
                            particle2.x
                            - particle1.x
                        )

                        dy = (
                            particle2.y
                            - particle1.y
                        )

                        distance2 = (
                            dx * dx
                            + dy * dy
                        )

                        if (
                            distance2
                            >= limit2
                        ):
                            continue

                        if (
                            distance2
                            < 1e-14
                        ):
                            angle = (
                                random.random()
                                * 2.0
                                * math.pi
                            )

                            nx = math.cos(
                                angle
                            )

                            ny = math.sin(
                                angle
                            )

                            distance = (
                                1e-7
                            )

                        else:
                            distance = (
                                math.sqrt(
                                    distance2
                                )
                            )

                            nx = (
                                dx
                                / distance
                            )

                            ny = (
                                dy
                                / distance
                            )

                        # --------------------------------------------------
                        # Position correction
                        # --------------------------------------------------

                        overlap = (
                            diameter
                            - distance
                        )

                        correction = (
                            0.5
                            * max(
                                0.0,
                                overlap,
                            )
                            + 1e-8
                        )

                        particle1.x -= (
                            correction
                            * nx
                        )

                        particle1.y -= (
                            correction
                            * ny
                        )

                        particle2.x += (
                            correction
                            * nx
                        )

                        particle2.y += (
                            correction
                            * ny
                        )

                        # --------------------------------------------------
                        # Elastic equal-mass impulse
                        # --------------------------------------------------

                        relative_normal_velocity = (
                            (
                                particle2.vx
                                - particle1.vx
                            )
                            * nx
                            + (
                                particle2.vy
                                - particle1.vy
                            )
                            * ny
                        )

                        # Only respond when particles approach each other.
                        if (
                            relative_normal_velocity
                            < 0.0
                        ):
                            impulse = (
                                -relative_normal_velocity
                            )

                            particle1.vx -= (
                                impulse
                                * nx
                            )

                            particle1.vy -= (
                                impulse
                                * ny
                            )

                            particle2.vx += (
                                impulse
                                * nx
                            )

                            particle2.vy += (
                                impulse
                                * ny
                            )

                        self._enforce_constraints(
                            particle1
                        )

                        self._enforce_constraints(
                            particle2
                        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _instant_histogram(
        self,
    ) -> list[float]:
        counts = [
            0
        ] * self.HIST_BINS

        if not self.particles:
            return [
                0.0
            ] * self.HIST_BINS

        for particle in self.particles:
            index = min(
                self.HIST_BINS - 1,
                max(
                    0,
                    int(
                        particle.x
                        * self.HIST_BINS
                    ),
                ),
            )

            counts[index] += 1

        inverse_count = (
            1.0
            / len(
                self.particles
            )
        )

        return [
            count
            * inverse_count
            for count in counts
        ]

    def _experimental_histogram(
        self,
        update: bool,
    ) -> list[float]:
        """
        Exponentially average the measured histogram.

        The data remain entirely experimental (particle positions), but mild
        temporal averaging makes comparison to the smooth theoretical curve
        much easier than an instantaneous 100-300 particle snapshot.
        """
        current = (
            self._instant_histogram()
        )

        if self.hist_ema is None:
            self.hist_ema = (
                current.copy()
            )

        elif update:
            alpha = 0.12

            self.hist_ema = [
                (
                    1.0 - alpha
                )
                * old
                + alpha
                * new
                for old, new in zip(
                    self.hist_ema,
                    current,
                )
            ]

        return (
            self.hist_ema.copy()
        )

    def _theoretical_histogram(
        self,
    ) -> list[float]:
        """
        Discretized ideal Boltzmann distribution

            P_i ∝ exp[-U(x_i) / T].

        Subtracting the minimum energy before exp() is mathematically harmless
        because the common factor cancels during normalization and improves
        numerical robustness.
        """
        temperature = max(
            1e-6,
            float(
                self.temp_var.get()
            ),
        )

        scale = float(
            self.energy_var.get()
        )

        energies = [
            scale
            * self.potential.sample(
                (
                    index + 0.5
                )
                / self.HIST_BINS
            )
            for index
            in range(
                self.HIST_BINS
            )
        ]

        minimum = min(
            energies
        )

        weights = [
            math.exp(
                -(
                    energy
                    - minimum
                )
                / temperature
            )
            for energy in energies
        ]

        total = sum(
            weights
        )

        return [
            weight
            / total
            for weight in weights
        ]

    def _center_of_mass(
        self,
    ) -> float:
        if not self.particles:
            return 0.0

        return (
            sum(
                particle.x
                for particle
                in self.particles
            )
            / len(
                self.particles
            )
        )

    def _entropy_x(
        self,
    ) -> float:
        """
        Coarse-grained configurational entropy of the x histogram,

            S_x = - sum p_i ln p_i,

        using k_B = 1.

        It should be interpreted as the entropy of the discretized positional
        distribution shown in the program, not as a complete thermodynamic
        entropy including velocities and correlations.
        """
        entropy = 0.0

        for probability in (
            self._instant_histogram()
        ):
            if probability > 0.0:
                entropy -= (
                    probability
                    * math.log(
                        probability
                    )
                )

        return entropy

    # ------------------------------------------------------------------
    # Potential rendering
    # ------------------------------------------------------------------

    def _redraw_potential(
        self,
    ) -> None:
        canvas = (
            self.potential_canvas
        )

        canvas.delete(
            "all"
        )

        width = self.BOX_W
        height = self.POT_H

        top = 12
        bottom = (
            height - 24
        )

        # Grid.
        for index in range(11):
            x = (
                index
                * width
                / 10
            )

            canvas.create_line(
                x,
                top,
                x,
                bottom,
                fill="#eeeeee",
            )

        for index in range(5):
            y = (
                top
                + index
                * (
                    bottom - top
                )
                / 4
            )

            canvas.create_line(
                0,
                y,
                width,
                y,
                fill="#eeeeee",
            )

        canvas.create_line(
            0,
            bottom,
            width,
            bottom,
            fill="#333333",
        )

        canvas.create_text(
            7,
            top + 2,
            anchor="nw",
            text=self.tr(
                "high_u"
            ),
            fill="#777777",
        )

        canvas.create_text(
            7,
            bottom - 2,
            anchor="sw",
            text=self.tr(
                "low_u"
            ),
            fill="#777777",
        )

        # Initial state is visibly blank.
        if self.potential.is_blank():
            canvas.create_text(
                width / 2,
                (
                    top + bottom
                ) / 2,
                text=self.tr(
                    "blank_hint"
                ),
                fill="#777777",
                font=(
                    "TkDefaultFont",
                    11,
                ),
            )

            return

        # --------------------------------------------------------------
        # Effective complete U(x): gray dashed line
        # --------------------------------------------------------------

        effective = (
            self.potential.compiled()
        )

        coordinates: list[
            float
        ] = []

        for index, value in enumerate(
            effective
        ):
            x = (
                index
                / (
                    self.potential.points
                    - 1
                )
                * width
            )

            y = (
                bottom
                - value
                * (
                    bottom - top
                )
            )

            coordinates.extend(
                (
                    x,
                    y,
                )
            )

        canvas.create_line(
            *coordinates,
            fill="#999999",
            width=2,
            dash=(5, 4),
        )

        # --------------------------------------------------------------
        # Explicitly drawn portions: solid red
        # --------------------------------------------------------------

        segment: list[
            float
        ] = []

        for index, value in enumerate(
            self.potential.values
        ):
            if value is None:
                if (
                    len(segment)
                    >= 4
                ):
                    canvas.create_line(
                        *segment,
                        fill="#c62828",
                        width=3,
                        smooth=True,
                    )

                segment = []
                continue

            x = (
                index
                / (
                    self.potential.points
                    - 1
                )
                * width
            )

            y = (
                bottom
                - value
                * (
                    bottom - top
                )
            )

            segment.extend(
                (
                    x,
                    y,
                )
            )

        if len(segment) >= 4:
            canvas.create_line(
                *segment,
                fill="#c62828",
                width=3,
                smooth=True,
            )

    # ------------------------------------------------------------------
    # Particle canvas rendering
    # ------------------------------------------------------------------

    def _redraw_partition_and_particles(
        self,
    ) -> None:
        canvas = (
            self.particle_canvas
        )

        canvas.delete(
            "all"
        )

        width = self.BOX_W
        height = self.BOX_H

        # Light x grid.
        for index in range(11):
            x = (
                index
                * width
                / 10
            )

            canvas.create_line(
                x,
                0,
                x,
                height,
                fill="#edf2f7",
            )

        # --------------------------------------------------------------
        # Partition
        # --------------------------------------------------------------

        if (
            self.partition_mode
            != "none"
        ):
            x = (
                float(
                    self.partition_x_var.get()
                )
                * width
            )

            if (
                self.partition_mode
                == "solid"
            ):
                canvas.create_line(
                    x,
                    0,
                    x,
                    height,
                    fill="#222222",
                    width=6,
                )

            else:
                pore = max(
                    2.0,
                    float(
                        self.pore_var.get()
                    ),
                )

                period = (
                    2.0 * pore
                )

                y = 0.0

                while y < height:
                    # first half of period = hole
                    # second half = material
                    start = (
                        y + pore
                    )

                    end = min(
                        y + period,
                        height,
                    )

                    if start < height:
                        canvas.create_line(
                            x,
                            start,
                            x,
                            end,
                            fill="#222222",
                            width=6,
                        )

                    y += period

        # --------------------------------------------------------------
        # Particle items
        # --------------------------------------------------------------

        radius = float(
            self.radius_var.get()
        )

        self.particle_items = []

        for particle in self.particles:
            x = (
                particle.x
                * width
            )

            # world y uses BOX_W as its physical-to-pixel scale
            y = (
                particle.y
                * self.BOX_W
            )

            item = (
                canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill="#4f9dd9",
                    outline="#24506b",
                    width=1,
                )
            )

            self.particle_items.append(
                item
            )

        # Center-of-mass line.
        center_x = (
            self._center_of_mass()
            * width
        )

        self.center_line_item = (
            canvas.create_line(
                center_x,
                0,
                center_x,
                height,
                fill="#f28e2b",
                width=2,
                dash=(5, 4),
            )
        )

    def _update_particle_items(
        self,
    ) -> None:
        """
        Reuse existing Canvas oval objects instead of deleting/recreating them
        every frame. This substantially reduces Tk/Tcl overhead at high N.
        """
        if (
            len(
                self.particle_items
            )
            != len(
                self.particles
            )
        ):
            self._redraw_partition_and_particles()
            return

        canvas = (
            self.particle_canvas
        )

        radius = float(
            self.radius_var.get()
        )

        for item, particle in zip(
            self.particle_items,
            self.particles,
        ):
            x = (
                particle.x
                * self.BOX_W
            )

            y = (
                particle.y
                * self.BOX_W
            )

            canvas.coords(
                item,
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            )

        if (
            self.center_line_item
            is not None
        ):
            center_x = (
                self._center_of_mass()
                * self.BOX_W
            )

            canvas.coords(
                self.center_line_item,
                center_x,
                0,
                center_x,
                self.BOX_H,
            )

    # ------------------------------------------------------------------
    # Histogram rendering
    # ------------------------------------------------------------------

    def _redraw_distribution(
        self,
        update_ema: bool,
    ) -> None:
        canvas = (
            self.distribution_canvas
        )

        canvas.delete(
            "all"
        )

        width = self.BOX_W
        height = self.DIST_H

        left = 43
        right = width - 12
        top = 24
        bottom = height - 30

        plot_width = (
            right - left
        )

        plot_height = (
            bottom - top
        )

        experimental = (
            self._experimental_histogram(
                update_ema
            )
        )

        theoretical = (
            self._theoretical_histogram()
        )

        maximum = max(
            max(
                experimental,
                default=0.01,
            ),
            max(
                theoretical,
                default=0.01,
            ),
            0.01,
        )

        # Grid.
        for index in range(5):
            y = (
                top
                + index
                * plot_height
                / 4
            )

            canvas.create_line(
                left,
                y,
                right,
                y,
                fill="#eeeeee",
            )

        bin_width = (
            plot_width
            / self.HIST_BINS
        )

        # --------------------------------------------------------------
        # Experimental bars
        # --------------------------------------------------------------

        for index, probability in enumerate(
            experimental
        ):
            x0 = (
                left
                + index
                * bin_width
                + 1
            )

            x1 = (
                left
                + (
                    index + 1
                )
                * bin_width
                - 1
            )

            y = (
                bottom
                - (
                    probability
                    / maximum
                )
                * plot_height
                * 0.92
            )

            canvas.create_rectangle(
                x0,
                y,
                x1,
                bottom,
                fill="#78aadd",
                outline="",
            )

        # --------------------------------------------------------------
        # Theoretical Boltzmann line
        # --------------------------------------------------------------

        points: list[
            float
        ] = []

        for index, probability in enumerate(
            theoretical
        ):
            x = (
                left
                + (
                    index + 0.5
                )
                * bin_width
            )

            y = (
                bottom
                - (
                    probability
                    / maximum
                )
                * plot_height
                * 0.92
            )

            points.extend(
                (
                    x,
                    y,
                )
            )

        if len(points) >= 4:
            canvas.create_line(
                *points,
                fill="#c62828",
                width=3,
                smooth=True,
            )

        # Axes.
        canvas.create_line(
            left,
            top,
            left,
            bottom,
            fill="#333333",
        )

        canvas.create_line(
            left,
            bottom,
            right,
            bottom,
            fill="#333333",
        )

        # Legend.
        canvas.create_rectangle(
            left + 10,
            7,
            left + 24,
            17,
            fill="#78aadd",
            outline="",
        )

        canvas.create_text(
            left + 30,
            12,
            anchor="w",
            text=self.tr(
                "experiment"
            ),
            fill="#333333",
        )

        canvas.create_line(
            left + 150,
            12,
            left + 174,
            12,
            fill="#c62828",
            width=3,
        )

        canvas.create_text(
            left + 182,
            12,
            anchor="w",
            text=self.tr(
                "theory"
            ),
            fill="#333333",
        )

        canvas.create_text(
            (
                left + right
            ) / 2,
            height - 10,
            text=self.tr(
                "x_axis"
            ),
            fill="#444444",
        )

        canvas.create_text(
            12,
            (
                top + bottom
            ) / 2,
            text="P(x)",
            angle=90,
            fill="#444444",
        )

    # ------------------------------------------------------------------
    # Statistics rendering
    # ------------------------------------------------------------------

    def _update_stats(
        self,
    ) -> None:
        state = (
            self.tr(
                "status_running"
            )
            if self.running
            else self.tr(
                "status_paused"
            )
        )

        self.stats_var.set(
            f"{self.tr('center_mass')}: "
            f"x = {self._center_of_mass():.3f}    "
            f"{self.tr('entropy')}: "
            f"{self._entropy_x():.3f}    "
            f"{self.tr('time')}: "
            f"{self.sim_time:.2f}    "
            f"[{state}]"
        )

    # ------------------------------------------------------------------
    # Main Tk animation loop
    # ------------------------------------------------------------------

    def _animation_loop(
        self,
    ) -> None:
        if self.running:
            for _ in range(
                self.STEPS_PER_FRAME
            ):
                self._simulation_step()

        self.frame_no += 1

        # Particle locations are lightweight enough to update every frame.
        self._update_particle_items()

        # Histograms and text need not be reconstructed 60 times per second.
        if (
            self.frame_no
            % 3
            == 0
        ):
            self._redraw_distribution(
                update_ema=self.running
            )

            self._update_stats()

        self.root.after(
            16,
            self._animation_loop,
        )


# -----------------------------------------------------------------------------
# Program entry point
# -----------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()

    BoltzmannLab(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
