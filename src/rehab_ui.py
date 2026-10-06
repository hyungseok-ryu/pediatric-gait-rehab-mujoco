"""소아 보행재활 MuJoCo 실시간 제어 UI.

실행::

    python src/rehab_ui.py

Tk UI와 MuJoCo viewer는 서로 다른 프로세스에서 실행된다. UI의 shared mapping을
시뮬레이션이 주기적으로 읽어 치료 모드, cadence, 관절별 보조량을 즉시 반영한다.
Active mode의 patient effort는 실제 센서가 연결되기 전 검증용 가상 입력이다.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import traceback
from datetime import datetime
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk


_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


BG = "#D8D8D8"
CARD = "#ECECEC"
INK = "#202020"
MUTED = "#555555"
NAVY = "#C5C5C5"
BLUE = "#4E6A7C"
BLUE_SOFT = "#D6DEE2"
GREEN = "#245B2A"
GREEN_SOFT = "#DCE7DC"
AMBER = "#745514"
AMBER_SOFT = "#E9E2D2"
RED = "#A52A2A"
RED_DARK = "#7F1D1D"
BORDER = "#8A8A8A"

FONT = "Noto Sans CJK KR"

STATE_LABELS = {
    "initializing": "모델 준비 중",
    "stand_hold": "기립 대기",
    "automatic": "Automatic · 자동 보행",
    "active": "Active · 능동 보행",
    "sit_to_stand": "Sit-to-Stand · 앉기-서기",
    "safe_stop": "안전 정지",
    "finished": "세션 종료",
}

ROBOT_LABELS = {
    "외골격형": "exoskeleton",
    "말단 구동형": "open_chain",
}


def _simulation_entry(control, kwargs: dict) -> None:
    """별도 프로세스에서 MuJoCo를 실행하고 오류를 UI로 전달한다."""
    try:
        from simulate import run_simulation

        run_simulation(runtime_control=control, **kwargs)
    except BaseException as exc:  # child-process boundary: surface every failure
        try:
            control["status"] = "error"
            control["actual_state"] = "error"
            control["error"] = f"{type(exc).__name__}: {exc}"
            control["traceback"] = traceback.format_exc()
        except Exception:
            pass
        raise


class RehabControlApp:
    """세 가지 재활 운동을 제공하는 연구용 보행재활 제어 화면."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("소아 보행 재활 제어 시스템")
        self.root.configure(bg=BG)
        self.root.geometry("1280x840")
        self.root.minsize(1160, 760)

        self.ctx = mp.get_context("spawn")
        self.manager = None
        self.control = None
        self.process: mp.Process | None = None
        self._closing = False
        self._alarm_shown = ""

        self.robot_var = tk.StringVar(value="외골격형")
        self.height_var = tk.DoubleVar(value=0.856)
        self.weight_var = tk.DoubleVar(value=12.0)
        self.minutes_var = tk.IntVar(value=3)
        self.cadence_var = tk.DoubleVar(value=40.0)
        self.soft_start_var = tk.DoubleVar(value=2.0)
        self.effort_var = tk.DoubleVar(value=0.0)
        self.effort_threshold_var = tk.DoubleVar(value=20.0)
        self.sit_to_stand_duration_var = tk.DoubleVar(value=3.0)
        self.phase_title_var = tk.StringVar(value="보행 위상")
        self.support_vars = {
            "right_hip_support": tk.DoubleVar(value=100.0),
            "right_knee_support": tk.DoubleVar(value=100.0),
            "left_hip_support": tk.DoubleVar(value=100.0),
            "left_knee_support": tk.DoubleVar(value=100.0),
        }
        # 현재 소아 보행 데이터(hip -10.1~44.2°, knee 5.8~76.3°)에
        # 약간의 여유를 둔 시뮬레이션 기본값. 실제 환자는 치료사가 설정한다.
        self.rom_vars = {
            "right_hip_min": tk.DoubleVar(value=-15.0),
            "right_hip_max": tk.DoubleVar(value=50.0),
            "right_knee_min": tk.DoubleVar(value=0.0),
            "right_knee_max": tk.DoubleVar(value=80.0),
            "left_hip_min": tk.DoubleVar(value=-15.0),
            "left_hip_max": tk.DoubleVar(value=50.0),
            "left_knee_min": tk.DoubleVar(value=0.0),
            "left_knee_max": tk.DoubleVar(value=80.0),
        }
        self.rom_summary_var = tk.StringVar()
        self._update_rom_summary()

        self.telemetry_vars = {
            "session": tk.StringVar(value="00:00"),
            "steps": tk.StringVar(value="0"),
            "phase": tk.StringVar(value="0 %"),
            "mode": tk.StringVar(value="대기"),
            "right_angles": tk.StringVar(value="-- / --°"),
            "left_angles": tk.StringVar(value="-- / --°"),
            "right_torque": tk.StringVar(value="-- / -- Nm"),
            "left_torque": tk.StringVar(value="-- / -- Nm"),
        }

        self._configure_styles()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_simulation)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "TCombobox",
            fieldbackground=CARD,
            background=CARD,
            foreground=INK,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=5,
            font=(FONT, 10),
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#BEBEBE",
            background=BLUE,
            bordercolor=BORDER,
            lightcolor=BLUE,
            darkcolor=BLUE,
        )

    def _build_layout(self) -> None:
        header = tk.Frame(
            self.root,
            bg=NAVY,
            height=58,
            relief="raised",
            bd=2,
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=NAVY)
        brand.pack(side="left", padx=16, pady=8)
        tk.Label(
            brand,
            text="소아 보행 재활 제어 시스템",
            bg=NAVY,
            fg=INK,
            font=(FONT, 16, "bold"),
        ).pack(anchor="w")

        status_wrap = tk.Frame(header, bg=NAVY)
        status_wrap.pack(side="right", padx=16)
        self.status_badge = tk.Label(
            status_wrap,
            text="시뮬레이션 준비",
            bg="#DADADA",
            fg=INK,
            padx=14,
            pady=6,
            relief="sunken",
            bd=2,
            font=(FONT, 9, "bold"),
        )
        self.status_badge.pack()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=12)
        body.grid_columnconfigure(0, weight=0, minsize=275)
        body.grid_columnconfigure(1, weight=2, minsize=470)
        body.grid_columnconfigure(2, weight=1, minsize=300)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        center = tk.Frame(body, bg=BG)
        right = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        center.grid(row=0, column=1, sticky="nsew", padx=5)
        right.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

        self._build_session_panel(left)
        self._build_control_panel(center)
        self._build_telemetry_panel(right)

    def _card(self, parent: tk.Widget, title: str, subtitle: str = "") -> tk.Frame:
        outer = tk.Frame(parent, bg=CARD, relief="groove", bd=2)
        outer.pack(fill="x", pady=(0, 9))
        head = tk.Frame(outer, bg=CARD)
        head.pack(fill="x", padx=12, pady=(10, 7))
        tk.Label(
            head, text=title, bg=CARD, fg=INK, font=(FONT, 11, "bold")
        ).pack(anchor="w")
        if subtitle:
            tk.Label(
                head,
                text=subtitle,
                bg=CARD,
                fg=MUTED,
                justify="left",
                wraplength=420,
                font=(FONT, 8),
            ).pack(anchor="w", pady=(2, 0))
        return outer

    def _field_label(self, parent: tk.Widget, text: str) -> None:
        tk.Label(
            parent, text=text, bg=CARD, fg=MUTED, font=(FONT, 9, "bold")
        ).pack(anchor="w", pady=(8, 4))

    def _update_rom_summary(self) -> None:
        right_hip = (
            self.rom_vars["right_hip_min"].get(),
            self.rom_vars["right_hip_max"].get(),
        )
        right_knee = (
            self.rom_vars["right_knee_min"].get(),
            self.rom_vars["right_knee_max"].get(),
        )
        left_hip = (
            self.rom_vars["left_hip_min"].get(),
            self.rom_vars["left_hip_max"].get(),
        )
        left_knee = (
            self.rom_vars["left_knee_min"].get(),
            self.rom_vars["left_knee_max"].get(),
        )
        self.rom_summary_var.set(
            f"오른쪽: 고관절 {right_hip[0]:.0f}~{right_hip[1]:.0f}°, "
            f"무릎 {right_knee[0]:.0f}~{right_knee[1]:.0f}°\n"
            f"왼쪽: 고관절 {left_hip[0]:.0f}~{left_hip[1]:.0f}°, "
            f"무릎 {left_knee[0]:.0f}~{left_knee[1]:.0f}°"
        )

    @staticmethod
    def _validate_rom_values(values: dict[str, float]) -> str | None:
        if not all(math.isfinite(value) for value in values.values()):
            return "모든 범위에 숫자를 입력해주세요."
        for side_label, side in (("오른쪽", "right"), ("왼쪽", "left")):
            hip_min = values[f"{side}_hip_min"]
            hip_max = values[f"{side}_hip_max"]
            knee_min = values[f"{side}_knee_min"]
            knee_max = values[f"{side}_knee_max"]
            if hip_min >= hip_max or knee_min >= knee_max:
                return f"{side_label} 관절의 최소각은 최대각보다 작아야 합니다."
            if not -90.0 <= hip_min < hip_max <= 120.0:
                return f"{side_label} 고관절 범위는 -90°~120° 안에서 설정해주세요."
            if not -10.0 <= knee_min < knee_max <= 150.0:
                return f"{side_label} 무릎 범위는 -10°~150° 안에서 설정해주세요."
            if not hip_min <= 0.0 <= hip_max or not knee_min <= 5.0 <= knee_max:
                return (
                    f"{side_label} 범위에 중립 기립자세 "
                    "(고관절 0°, 무릎 5°)가 포함되어야 합니다."
                )
        return None

    def _open_rom_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("관절 구동 범위 설정")
        dialog.configure(bg=BG)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        tk.Label(
            dialog,
            text="관절 구동 범위 (단위: 도)",
            bg=BG,
            fg=INK,
            font=(FONT, 11, "bold"),
        ).grid(row=0, column=0, columnspan=5, sticky="w", padx=14, pady=(12, 4))
        tk.Label(
            dialog,
            text="세션 시작 전에 환자별 범위를 설정합니다.",
            bg=BG,
            fg=MUTED,
            font=(FONT, 8),
        ).grid(row=1, column=0, columnspan=5, sticky="w", padx=14, pady=(0, 10))

        headers = ("관절", "오른쪽 최소", "오른쪽 최대", "왼쪽 최소", "왼쪽 최대")
        for column, text_value in enumerate(headers):
            tk.Label(
                dialog,
                text=text_value,
                bg="#C9C9C9",
                fg=INK,
                relief="ridge",
                bd=1,
                padx=8,
                pady=5,
                font=(FONT, 9, "bold"),
            ).grid(row=2, column=column, sticky="nsew", padx=1, pady=1)

        temp_vars = {
            key: tk.DoubleVar(value=variable.get())
            for key, variable in self.rom_vars.items()
        }
        rows = (
            ("고관절", "hip", -90.0, 120.0),
            ("무릎", "knee", -10.0, 150.0),
        )
        for row_index, (label, joint, minimum, maximum) in enumerate(rows, start=3):
            tk.Label(
                dialog,
                text=label,
                bg=CARD,
                fg=INK,
                relief="ridge",
                bd=1,
                padx=8,
                pady=6,
                font=(FONT, 9),
            ).grid(row=row_index, column=0, sticky="nsew", padx=1, pady=1)
            keys = (
                f"right_{joint}_min",
                f"right_{joint}_max",
                f"left_{joint}_min",
                f"left_{joint}_max",
            )
            for column, key in enumerate(keys, start=1):
                tk.Spinbox(
                    dialog,
                    from_=minimum,
                    to=maximum,
                    increment=1.0,
                    textvariable=temp_vars[key],
                    width=10,
                    justify="right",
                    relief="sunken",
                    bd=1,
                    font=(FONT, 9),
                ).grid(row=row_index, column=column, sticky="ew", padx=1, pady=1)

        note = tk.Label(
            dialog,
            text=(
                "기본값: 고관절 -15~50°, 무릎 0~80°\n"
                "실제 환자 적용 범위는 치료사가 결정해야 합니다."
            ),
            bg=BG,
            fg=MUTED,
            justify="left",
            font=(FONT, 8),
        )
        note.grid(row=5, column=0, columnspan=5, sticky="w", padx=14, pady=(10, 8))

        buttons = tk.Frame(dialog, bg=BG)
        buttons.grid(row=6, column=0, columnspan=5, sticky="e", padx=12, pady=(0, 12))

        def apply_values() -> None:
            try:
                values = {key: float(variable.get()) for key, variable in temp_vars.items()}
            except (tk.TclError, ValueError):
                messagebox.showerror("입력 오류", "관절 범위에 숫자를 입력해주세요.", parent=dialog)
                return
            error = self._validate_rom_values(values)
            if error:
                messagebox.showerror("범위 설정 오류", error, parent=dialog)
                return
            for key, value in values.items():
                self.rom_vars[key].set(value)
            self._update_rom_summary()
            dialog.destroy()

        tk.Button(
            buttons,
            text="적용",
            command=apply_values,
            width=9,
            relief="raised",
            bd=2,
            font=(FONT, 9),
        ).pack(side="left", padx=3)
        tk.Button(
            buttons,
            text="취소",
            command=dialog.destroy,
            width=9,
            relief="raised",
            bd=2,
            font=(FONT, 9),
        ).pack(side="left", padx=3)

        dialog.grab_set()
        dialog.wait_visibility()
        dialog.focus_set()

    def _build_session_panel(self, parent: tk.Frame) -> None:
        card = self._card(parent, "세션 설정", "환자 기본 정보와 로봇 구조")
        content = tk.Frame(card, bg=CARD)
        content.pack(fill="x", padx=18, pady=(0, 18))

        self._field_label(content, "로봇 구조")
        self.robot_combo = ttk.Combobox(
            content,
            textvariable=self.robot_var,
            values=tuple(ROBOT_LABELS),
            state="readonly",
        )
        self.robot_combo.pack(fill="x")

        self._field_label(content, "관절 구동 범위")
        self.rom_button = tk.Button(
            content,
            text="좌우 관절 범위 설정...",
            command=self._open_rom_dialog,
            bg="#D0D0D0",
            activebackground="#BEBEBE",
            fg=INK,
            relief="raised",
            bd=2,
            pady=5,
            font=(FONT, 9),
        )
        self.rom_button.pack(fill="x")
        tk.Label(
            content,
            textvariable=self.rom_summary_var,
            bg=CARD,
            fg=MUTED,
            justify="left",
            font=(FONT, 8),
        ).pack(anchor="w", pady=(3, 0))

        measurements = tk.Frame(content, bg=CARD)
        measurements.pack(fill="x", pady=(5, 0))
        measurements.grid_columnconfigure(0, weight=1)
        measurements.grid_columnconfigure(1, weight=1)

        height_box = tk.Frame(measurements, bg=CARD)
        height_box.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self._field_label(height_box, "키 (m)")
        self.height_spin = tk.Spinbox(
            height_box,
            from_=0.60,
            to=1.30,
            increment=0.001,
            textvariable=self.height_var,
            font=(FONT, 10),
            relief="solid",
            bd=1,
            highlightthickness=0,
        )
        self.height_spin.pack(fill="x", ipady=7)

        weight_box = tk.Frame(measurements, bg=CARD)
        weight_box.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self._field_label(weight_box, "체중 (kg)")
        self.weight_spin = tk.Spinbox(
            weight_box,
            from_=5.0,
            to=40.0,
            increment=0.1,
            textvariable=self.weight_var,
            font=(FONT, 10),
            relief="solid",
            bd=1,
            highlightthickness=0,
        )
        self.weight_spin.pack(fill="x", ipady=7)

        self._field_label(content, "세션 길이 (분)")
        self.minutes_spin = tk.Spinbox(
            content,
            from_=1,
            to=30,
            increment=1,
            textvariable=self.minutes_var,
            font=(FONT, 10),
            relief="solid",
            bd=1,
            highlightthickness=0,
        )
        self.minutes_spin.pack(fill="x", ipady=7)

        self.launch_button = tk.Button(
            content,
            text="시뮬레이션 시작",
            command=self._start_session,
            bg="#D0D0D0",
            activebackground="#BEBEBE",
            fg=INK,
            relief="raised",
            bd=2,
            pady=8,
            font=(FONT, 10, "bold"),
        )
        self.launch_button.pack(fill="x", pady=(18, 7))
        self.end_button = tk.Button(
            content,
            text="세션 종료",
            command=self._stop_session,
            bg="#E9EDF2",
            activebackground="#DDE3EA",
            fg=INK,
            relief="raised",
            bd=2,
            pady=7,
            state="disabled",
            font=(FONT, 10, "bold"),
        )
        self.end_button.pack(fill="x")

        note = tk.Frame(parent, bg="#E2E2E2", relief="groove", bd=2)
        note.pack(fill="x")
        tk.Label(
            note,
            text="연구용 시뮬레이션",
            bg="#E2E2E2",
            fg=INK,
            font=(FONT, 10, "bold"),
        ).pack(anchor="w", padx=15, pady=(13, 3))
        tk.Label(
            note,
            text=(
                "실제 환자 적용 전 임상가가 관절 가동범위와 토크 한계를 "
                "환자별로 검토해야 합니다."
            ),
            bg="#E2E2E2",
            fg=MUTED,
            wraplength=235,
            justify="left",
            font=(FONT, 9),
        ).pack(anchor="w", padx=15, pady=(0, 13))

    def _build_control_panel(self, parent: tk.Frame) -> None:
        modes = self._card(
            parent,
            "재활 제어 모드",
            "자동 보행, 환자 힘 기반 진행 또는 앉기-서기를 선택하세요.",
        )
        mode_content = tk.Frame(modes, bg=CARD)
        mode_content.pack(fill="x", padx=18, pady=(0, 18))
        for column in range(3):
            mode_content.grid_columnconfigure(column, weight=1)

        self.mode_buttons: dict[str, tk.Button] = {}
        mode_specs = (
            ("automatic", "Automatic", "자동 보행"),
            ("active", "Active", "힘 기준 진행"),
            ("sit_to_stand", "Sit-to-Stand", "앉기-서기 1회"),
        )
        for col, (mode, title, desc) in enumerate(mode_specs):
            button = tk.Button(
                mode_content,
                text=f"{title}\n{desc}",
                command=lambda m=mode: self._set_mode(m),
                bg="#D8D8D8",
                activebackground="#C0C0C0",
                fg=INK,
                relief="raised",
                bd=2,
                height=3,
                wraplength=125,
                font=(FONT, 9, "bold"),
            )
            button.grid(row=0, column=col, sticky="ew", padx=4)
            self.mode_buttons[mode] = button

        self.safe_button = tk.Button(
            mode_content,
            text="안전 정지",
            command=lambda: self._set_mode("safe_stop"),
            bg=RED,
            activebackground=RED_DARK,
            fg="white",
            activeforeground="white",
            relief="raised",
            bd=3,
            pady=9,
            font=(FONT, 11, "bold"),
        )
        self.safe_button.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=(10, 0))

        gait = self._card(parent, "제어 설정", "보행 속도, 동작 시간과 좌우 관절 보조율")
        gait_content = tk.Frame(gait, bg=CARD)
        gait_content.pack(fill="both", expand=True, padx=18, pady=(0, 16))

        self._scale_row(
            gait_content,
            "보행 속도",
            self.cadence_var,
            10,
            70,
            " 걸음/분",
            self._control_changed,
        )
        self._scale_row(
            gait_content,
            "부드러운 시작 시간",
            self.soft_start_var,
            0.5,
            5.0,
            " s",
            self._control_changed,
            compact=True,
            decimals=1,
            resolution=0.1,
        )
        self._scale_row(
            gait_content,
            "앉기-서기 상승 시간",
            self.sit_to_stand_duration_var,
            1.5,
            8.0,
            " s",
            self._control_changed,
            compact=True,
            decimals=1,
            resolution=0.1,
        )

        divider = tk.Frame(gait_content, bg=BORDER, height=1)
        divider.pack(fill="x", pady=10)

        support_grid = tk.Frame(gait_content, bg=CARD)
        support_grid.pack(fill="x")
        support_grid.grid_columnconfigure(0, weight=1)
        support_grid.grid_columnconfigure(1, weight=1)
        for col, side in enumerate(("right", "left")):
            side_frame = tk.Frame(support_grid, bg=CARD)
            side_frame.grid(row=0, column=col, sticky="nsew", padx=(0, 8) if col == 0 else (8, 0))
            tk.Label(
                side_frame,
                text="오른쪽" if side == "right" else "왼쪽",
                bg=CARD,
                fg=INK,
                font=(FONT, 10, "bold"),
            ).pack(anchor="w")
            self._scale_row(
                side_frame,
                "고관절 보조율",
                self.support_vars[f"{side}_hip_support"],
                0,
                100,
                " %",
                self._control_changed,
                compact=True,
            )
            self._scale_row(
                side_frame,
                "무릎 보조율",
                self.support_vars[f"{side}_knee_support"],
                0,
                100,
                " %",
                self._control_changed,
                compact=True,
            )

        effort_box = tk.Frame(
            gait_content,
            bg=AMBER_SOFT,
            highlightbackground="#F2D9AD",
            highlightthickness=1,
        )
        effort_box.pack(fill="x", pady=(13, 0))
        tk.Label(
            effort_box,
            text="Active mode 시험 입력",
            bg=AMBER_SOFT,
            fg=AMBER,
            font=(FONT, 9, "bold"),
        ).pack(anchor="w", padx=12, pady=(9, 0))
        effort_inner = tk.Frame(effort_box, bg=AMBER_SOFT)
        effort_inner.pack(fill="x", padx=12, pady=(0, 8))
        self._scale_row(
            effort_inner,
            "환자 힘 입력",
            self.effort_var,
            0,
            100,
            " %",
            self._control_changed,
            compact=True,
            background=AMBER_SOFT,
        )
        self._scale_row(
            effort_inner,
            "진행 기준",
            self.effort_threshold_var,
            5,
            80,
            " %",
            self._control_changed,
            compact=True,
            background=AMBER_SOFT,
        )

    def _scale_row(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
        unit: str,
        command,
        compact: bool = False,
        background: str = CARD,
        decimals: int = 0,
        resolution: float = 1.0,
    ) -> None:
        row = tk.Frame(parent, bg=background)
        row.pack(fill="x", pady=(5, 2) if compact else (7, 5))
        top = tk.Frame(row, bg=background)
        top.pack(fill="x")
        tk.Label(
            top,
            text=label,
            bg=background,
            fg=MUTED,
            font=(FONT, 8 if compact else 9, "bold"),
        ).pack(side="left")
        value_label = tk.Label(
            top,
            text=f"{variable.get():.{decimals}f}{unit}",
            bg=background,
            fg=INK,
            font=(FONT, 9, "bold"),
        )
        value_label.pack(side="right")

        def changed(raw: str) -> None:
            value_label.configure(
                text=f"{float(raw):.{decimals}f}{unit}"
            )
            command()

        tk.Scale(
            row,
            from_=minimum,
            to=maximum,
            orient="horizontal",
            showvalue=False,
            variable=variable,
            command=changed,
            bg=background,
            fg=INK,
            troughcolor="#BDBDBD",
            activebackground=BLUE,
            highlightthickness=0,
            bd=0,
            sliderlength=16,
            width=7,
            resolution=resolution,
        ).pack(fill="x")

    def _build_telemetry_panel(self, parent: tk.Frame) -> None:
        card = self._card(parent, "실시간 상태", "시뮬레이션 측정값")
        content = tk.Frame(card, bg=CARD)
        content.pack(fill="x", padx=18, pady=(0, 18))

        self.mode_banner = tk.Label(
            content,
            textvariable=self.telemetry_vars["mode"],
            bg="#D7D7D7",
            fg=INK,
            pady=12,
            font=(FONT, 12, "bold"),
        )
        self.mode_banner.pack(fill="x", pady=(0, 13))

        metrics = tk.Frame(content, bg=CARD)
        metrics.pack(fill="x")
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_columnconfigure(1, weight=1)
        self._metric(metrics, 0, 0, "세션 시간", self.telemetry_vars["session"])
        self._metric(metrics, 0, 1, "걸음 / 반복", self.telemetry_vars["steps"])

        tk.Label(
            content,
            textvariable=self.phase_title_var,
            bg=CARD,
            fg=MUTED,
            font=(FONT, 8, "bold"),
        ).pack(anchor="w", pady=(16, 4))
        phase_line = tk.Frame(content, bg=CARD)
        phase_line.pack(fill="x")
        self.phase_bar = ttk.Progressbar(
            phase_line, maximum=100, value=0, style="Horizontal.TProgressbar"
        )
        self.phase_bar.pack(side="left", fill="x", expand=True)
        tk.Label(
            phase_line,
            textvariable=self.telemetry_vars["phase"],
            width=6,
            anchor="e",
            bg=CARD,
            fg=INK,
            font=(FONT, 9, "bold"),
        ).pack(side="right")

        divider = tk.Frame(content, bg=BORDER, height=1)
        divider.pack(fill="x", pady=16)

        self._telemetry_row(content, "오른쪽 각도  고관절 / 무릎", "right_angles")
        self._telemetry_row(content, "왼쪽 각도  고관절 / 무릎", "left_angles")
        self._telemetry_row(content, "오른쪽 토크  고관절 / 무릎", "right_torque")
        self._telemetry_row(content, "왼쪽 토크  고관절 / 무릎", "left_torque")

        self.alarm_box = tk.Label(
            parent,
            text="안전 상태 정상\n추종 오차 감시 중",
            bg=GREEN_SOFT,
            fg=GREEN,
            justify="left",
            anchor="w",
            padx=16,
            pady=13,
            font=(FONT, 10, "bold"),
            highlightbackground="#BFE6D3",
            highlightthickness=1,
        )
        self.alarm_box.pack(fill="x")

        tk.Label(
            parent,
            text=(
                "Active mode의 환자 힘 입력은 센서 대신 사용하는 시험값입니다. "
                "실제 장치에서는 검증된 힘·토크 센서와 안전 조건이 필요합니다."
            ),
            bg=BG,
            fg=MUTED,
            wraplength=300,
            justify="left",
            font=(FONT, 8),
        ).pack(anchor="w", pady=(13, 0))

    def _metric(
        self,
        parent: tk.Widget,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        box = tk.Frame(parent, bg="#DEDEDE", relief="sunken", bd=1)
        box.grid(row=row, column=column, sticky="nsew", padx=(0, 5) if column == 0 else (5, 0))
        tk.Label(
            box, text=label, bg="#DEDEDE", fg=MUTED, font=(FONT, 8, "bold")
        ).pack(anchor="w", padx=11, pady=(9, 0))
        tk.Label(
            box, textvariable=variable, bg="#DEDEDE", fg=INK, font=(FONT, 17, "bold")
        ).pack(anchor="w", padx=11, pady=(0, 9))

    def _telemetry_row(self, parent: tk.Widget, label: str, key: str) -> None:
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, bg=CARD, fg=MUTED, font=(FONT, 8)).pack(anchor="w")
        tk.Label(
            row,
            textvariable=self.telemetry_vars[key],
            bg=CARD,
            fg=INK,
            font=(FONT, 11, "bold"),
        ).pack(anchor="w")

    def _initial_control(self) -> dict:
        values = {
            "status": "initializing",
            "actual_state": "initializing",
            "requested_state": "stand_hold",
            "stop_requested": False,
            "cadence_spm": float(self.cadence_var.get()),
            "soft_start_duration": float(self.soft_start_var.get()),
            "patient_effort": float(self.effort_var.get()),
            "initiation_threshold": float(self.effort_threshold_var.get()),
            "sit_to_stand_duration": float(self.sit_to_stand_duration_var.get()),
            "sit_to_stand_request_id": 0,
            "alarm": "",
        }
        values.update({key: float(var.get()) for key, var in self.support_vars.items()})
        return values

    def _start_session(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        try:
            height = float(self.height_var.get())
            weight = float(self.weight_var.get())
            minutes = int(self.minutes_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("입력 오류", "키, 체중, 세션 길이를 확인해주세요.")
            return
        if not 0.5 <= height <= 1.5 or not 3.0 <= weight <= 60.0 or not 1 <= minutes <= 30:
            messagebox.showerror("입력 범위 오류", "환자 정보와 세션 길이의 입력 범위를 확인해주세요.")
            return
        try:
            rom_values = {
                key: float(variable.get())
                for key, variable in self.rom_vars.items()
            }
        except (tk.TclError, ValueError):
            messagebox.showerror("입력 오류", "관절 구동 범위를 확인해주세요.")
            return
        rom_error = self._validate_rom_values(rom_values)
        if rom_error:
            messagebox.showerror("관절 범위 오류", rom_error)
            return

        self.manager = self.ctx.Manager()
        self.control = self.manager.dict(self._initial_control())
        timestamp = datetime.now().strftime("ui_session_%Y%m%d_%H%M%S")
        kwargs = {
            "render_mode": "viewer",
            # 기본 궤적이 1초/주기이므로 cycles가 곧 최대 실행 초 수이다.
            "n_cycles": max(1, minutes * 60),
            "height": height,
            "weight": weight,
            "robot_type": ROBOT_LABELS[self.robot_var.get()],
            "out_dir_name": timestamp,
            "save_video": False,
            "camera_name": "presentation",
            "right_hip_range_deg": (
                rom_values["right_hip_min"],
                rom_values["right_hip_max"],
            ),
            "right_knee_range_deg": (
                rom_values["right_knee_min"],
                rom_values["right_knee_max"],
            ),
            "left_hip_range_deg": (
                rom_values["left_hip_min"],
                rom_values["left_hip_max"],
            ),
            "left_knee_range_deg": (
                rom_values["left_knee_min"],
                rom_values["left_knee_max"],
            ),
        }
        self.process = self.ctx.Process(
            target=_simulation_entry,
            args=(self.control, kwargs),
            daemon=True,
        )
        self.process.start()
        self._set_session_inputs_enabled(False)
        self.launch_button.configure(state="disabled", text="모델 준비 중...")
        self.end_button.configure(state="normal")
        self._set_status("initializing")
        self._set_mode_visual("stand_hold")

    def _stop_session(self) -> None:
        if self.control is not None:
            try:
                self.control["stop_requested"] = True
                self.control["requested_state"] = "safe_stop"
            except (BrokenPipeError, ConnectionError, EOFError):
                pass
        self.end_button.configure(state="disabled", text="종료 중…")

    def _set_mode(self, mode: str) -> None:
        if self.control is None or self.process is None or not self.process.is_alive():
            messagebox.showinfo("세션 필요", "먼저 시뮬레이션을 시작해주세요.")
            return
        actual = str(self.control.get("actual_state", ""))
        if actual == "safe_stop" and mode != "safe_stop":
            if not messagebox.askyesno(
                "안전정지 해제",
                "알람 원인을 확인했습니까? 안전정지를 해제하고 기립 유지로 전환합니다.",
            ):
                return
            mode = "stand_hold"
            self.control["alarm"] = ""
            self._alarm_shown = ""
        self.control["requested_state"] = mode
        if mode == "sit_to_stand":
            self.control["sit_to_stand_request_id"] = int(
                self.control.get("sit_to_stand_request_id", 0)
            ) + 1
        self._set_mode_visual(mode)

    def _set_mode_visual(self, active: str) -> None:
        for mode, button in self.mode_buttons.items():
            selected = mode == active
            button.configure(
                bg="#C8D2D8" if selected else "#D8D8D8",
                fg=INK,
                relief="sunken" if selected else "raised",
            )
        if active == "safe_stop":
            self.mode_banner.configure(bg="#FCECEC", fg=RED)
        elif active in {"automatic", "active", "sit_to_stand"}:
            self.mode_banner.configure(bg="#C8D2D8", fg=INK)
        else:
            self.mode_banner.configure(bg="#D7D7D7", fg=INK)

    def _control_changed(self) -> None:
        if self.control is None:
            return
        try:
            self.control["cadence_spm"] = float(self.cadence_var.get())
            self.control["soft_start_duration"] = float(
                self.soft_start_var.get()
            )
            self.control["patient_effort"] = float(self.effort_var.get())
            self.control["initiation_threshold"] = float(
                self.effort_threshold_var.get()
            )
            self.control["sit_to_stand_duration"] = float(
                self.sit_to_stand_duration_var.get()
            )
            for key, var in self.support_vars.items():
                self.control[key] = float(var.get())
        except (BrokenPipeError, ConnectionError, EOFError, tk.TclError):
            pass

    def _set_session_inputs_enabled(self, enabled: bool) -> None:
        combo_state = "readonly" if enabled else "disabled"
        spin_state = "normal" if enabled else "disabled"
        self.robot_combo.configure(state=combo_state)
        self.height_spin.configure(state=spin_state)
        self.weight_spin.configure(state=spin_state)
        self.minutes_spin.configure(state=spin_state)
        self.rom_button.configure(state=spin_state)

    def _set_status(self, status: str) -> None:
        if status == "running":
            self.status_badge.configure(
                text="시뮬레이션 실행 중", bg="#C8D9C8", fg=GREEN
            )
        elif status == "initializing":
            self.status_badge.configure(
                text="모델 준비 중", bg="#DED4BC", fg=AMBER
            )
        elif status == "error":
            self.status_badge.configure(
                text="실행 오류", bg="#E4C7C7", fg=RED
            )
        else:
            self.status_badge.configure(
                text="시뮬레이션 준비", bg="#DADADA", fg=INK
            )

    @staticmethod
    def _format_time(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"

    def _poll_simulation(self) -> None:
        if self._closing:
            return
        if self.control is not None:
            try:
                status = str(self.control.get("status", "initializing"))
                actual = str(self.control.get("actual_state", "initializing"))
                self._set_status(status)
                mode_label = STATE_LABELS.get(actual, actual.upper())
                if actual == "active" and bool(
                    self.control.get("active_gate_blocked", False)
                ):
                    mode_label = "Active · 환자 힘 입력 대기"
                elif actual == "sit_to_stand":
                    stage = str(self.control.get("sit_to_stand_stage", ""))
                    stage_labels = {
                        "preparing": "앉은 자세 준비",
                        "seated_hold": "앉은 자세 유지",
                        "rising": "일어서기",
                        "complete": "기립 완료",
                    }
                    if stage in stage_labels:
                        mode_label = f"Sit-to-Stand · {stage_labels[stage]}"
                self.telemetry_vars["mode"].set(mode_label)
                self._set_mode_visual(actual)

                sim_time = float(self.control.get("sim_time", 0.0))
                phase = float(self.control.get("gait_phase_pct", 0.0))
                self.telemetry_vars["session"].set(self._format_time(sim_time))
                count = (
                    int(self.control.get("sit_to_stand_repetitions", 0))
                    if actual == "sit_to_stand"
                    else int(self.control.get("step_count", 0))
                )
                self.telemetry_vars["steps"].set(str(count))
                self.phase_title_var.set(
                    "동작 진행" if actual == "sit_to_stand" else "보행 위상"
                )
                self.telemetry_vars["phase"].set(f"{phase:.0f} %")
                self.phase_bar["value"] = phase

                rh = float(self.control.get("right_hip_deg", 0.0))
                rk = float(self.control.get("right_knee_deg", 0.0))
                lh = float(self.control.get("left_hip_deg", 0.0))
                lk = float(self.control.get("left_knee_deg", 0.0))
                rt1 = float(self.control.get("right_hip_torque", 0.0))
                rt2 = float(self.control.get("right_knee_torque", 0.0))
                lt1 = float(self.control.get("left_hip_torque", 0.0))
                lt2 = float(self.control.get("left_knee_torque", 0.0))
                self.telemetry_vars["right_angles"].set(f"{rh:5.1f} / {rk:5.1f}°")
                self.telemetry_vars["left_angles"].set(f"{lh:5.1f} / {lk:5.1f}°")
                self.telemetry_vars["right_torque"].set(f"{rt1:5.1f} / {rt2:5.1f} Nm")
                self.telemetry_vars["left_torque"].set(f"{lt1:5.1f} / {lt2:5.1f} Nm")

                alarm = str(self.control.get("alarm", ""))
                if alarm:
                    self.alarm_box.configure(
                        text=f"안전정지 활성\n{alarm}", bg="#FCECEC", fg=RED
                    )
                    if alarm != self._alarm_shown:
                        self._alarm_shown = alarm
                        messagebox.showwarning("안전정지", alarm)
                else:
                    self.alarm_box.configure(
                        text="안전 상태 정상\n추종 오차 감시 중",
                        bg=GREEN_SOFT,
                        fg=GREEN,
                    )

                if status == "error":
                    error = str(self.control.get("error", "알 수 없는 오류"))
                    if error != self._alarm_shown:
                        self._alarm_shown = error
                        messagebox.showerror("시뮬레이션 실행 오류", error)
            except (BrokenPipeError, ConnectionError, EOFError, tk.TclError):
                pass

        if self.process is not None and not self.process.is_alive():
            self.process.join(timeout=0.1)
            self.process = None
            was_error = bool(
                self.control is not None
                and self.control.get("status", "") == "error"
            )
            self._set_session_inputs_enabled(True)
            self.launch_button.configure(state="normal", text="시뮬레이션 시작")
            self.end_button.configure(state="disabled", text="세션 종료")
            if not was_error:
                self._set_status("finished")
            if self.manager is not None:
                try:
                    self.manager.shutdown()
                except Exception:
                    pass
            self.manager = None
            self.control = None

        self.root.after(100, self._poll_simulation)

    def _on_close(self) -> None:
        self._closing = True
        if self.control is not None:
            try:
                self.control["stop_requested"] = True
            except Exception:
                pass
        if self.process is not None and self.process.is_alive():
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        if self.manager is not None:
            try:
                self.manager.shutdown()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    mp.freeze_support()
    root = tk.Tk()
    RehabControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
