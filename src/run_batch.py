"""
run_batch.py
============
소아 보행 보조 로봇 MuJoCo 배치 시뮬레이션 실행 스크립트.

목적
----
- gait_data.csv 안의 모든 (Subject, Speed) 조합을 대상으로 테스트
- exoskeleton / open_chain 구조 비교
- 소아 체형, open-chain 링크 길이, PD gain 조건을 sweep
- 각 run의 성공/실패와 핵심 성능 지표를 batch_summary.csv로 저장

실행 위치
---------
프로젝트 루트(simulation_mujoco)에서 실행:

    python src/run_batch.py

대표 옵션
---------
    # 전체 조건 실행
    python src/run_batch.py

    # 먼저 실행 목록만 확인
    python src/run_batch.py --dry-run

    # 빠른 확인용으로 앞 20개만 실행
    python src/run_batch.py --max-runs 20

    # 이미 torque_profile.csv가 있는 조건은 건너뛰기
    python src/run_batch.py --skip-existing

    # 여러 조건을 동시에 실행
    python src/run_batch.py --workers 4

    # 영상까지 저장하고 싶을 때. 전체 batch에서는 매우 느릴 수 있음.
    python src/run_batch.py --render offscreen --cycles 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import shlex
import subprocess
import sys
import time
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import KOREAN_TODDLER_CASES


# ─────────────────────────────────────────────────────────────────────────────
# 기본 경로
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
save_dir = "batch_results"
DATA_CSV = ROOT / "data" / "gait_data.csv"
BATCH_ROOT = ROOT / "outputs" / save_dir
SUMMARY_CSV = ROOT / "outputs" / "batch_summary.csv"
FAILED_LOG = ROOT / "outputs" / "batch_failed_runs.txt"
SPEED_MATCH_ATOL = 5e-4


# ─────────────────────────────────────────────────────────────────────────────
# 조건 정의
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_TYPES = [
    "exoskeleton",
    "open_chain",
]

# PD gain 조건.
# 기존처럼 "kp", "kd"만 쓰면 두 모터에 같은 값이 적용된다.
# 모터별로 다르게 쓰려면 아래처럼 kp_motor1/kd_motor1/kp_motor2/kd_motor2를 지정한다.
PD_GAINS = [
    # {"name": "Kp100",  "kp": 100.0,  "kd": 1.0},
    {
        "name": "M1Kp1500Kd3_M2Kp1500Kd3",
        "kp_motor1": 1500.0,
        "kd_motor1": 3.0,
        "kp_motor2": 1500.0,
        "kd_motor2": 3.0,
    },
]

# 한국 소아 대표 체형: 12개월 여아 P3, KIPGroS 22–23개월 남녀 P50 평균,
# 35개월 남아 P97. 값의 단일 원본은 config.py에서 관리한다.
ANTHRO_CASES = [
    {"name": case.name, "height": case.height, "weight": case.weight}
    for case in KOREAN_TODDLER_CASES
]


REMOTE_A_VALUES = [0.11, 0.13, 0.15, 0.17]  # motor1 -> motor2 horizontal distance
REMOTE_B_VALUES = [0.02, 0.04, 0.06, 0.08]  # motor1 -> motor2 vertical distance
REMOTE_C_VALUES = [0.07, 0.09, 0.11, 0.13]  # motor2 crank/rod common length
REMOTE_D_VALUES = [0.06, 0.07, 0.08, 0.09]  # AE/BE upper plate link length
REMOTE_E_VALUES = [0.07, 0.09, 0.11, 0.13]  # AB / CD short spacing
REMOTE_F_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]  # AD / BC long link length
# REMOTE_G_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]  # distal link2 length
REMOTE_G_VALUES = [0.15, 0.20, 0.25]

def format_length_mm(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def make_arm_cases(robot_type: str) -> list[dict]:
    if robot_type == "exoskeleton":
        # 외골격 링크 길이는 각 대상자의 thigh/shank 길이에 자동 정렬된다.
        return [{"name": "HipKneeAligned"}]

    if robot_type == "open_chain":
        # 각 대상자의 thigh/shank 길이에 도달 여유율을 적용해 자동 산정한다.
        return [{"name": "OpenChain_AutoSized"}]

    if robot_type == "remote_exoskeleton":
        # 상부 전달기구 기본값을 사용하고 링크는 대상자 thigh/shank에 정렬한다.
        return [{"name": "RemoteHipKneeAligned"}]

    if robot_type == "remote_ankle_arm":
        # 사람 분절과 독립적인 기본 350 mm + 350 mm 발목 추종 팔.
        return [{
            "name": "AnkleArm_L350_L350",
            "link1": 0.35,
            "link2": 0.35,
        }]

    if robot_type == "remote_parallelogram":
        cases = []
        for remote_a in REMOTE_A_VALUES:
            for remote_b in REMOTE_B_VALUES:
                for remote_c in REMOTE_C_VALUES:
                    for remote_d in REMOTE_D_VALUES:
                        for remote_e in REMOTE_E_VALUES:
                            # A-E=d and B-E=d require d >= e/2.
                            if remote_d < 0.5 * remote_e:
                                continue
                            for remote_f in REMOTE_F_VALUES:
                                for remote_g in REMOTE_G_VALUES:
                                    cases.append({
                                        "name": (
                                            f"Arm_A{format_length_mm(remote_a)}_B{format_length_mm(remote_b)}_"
                                            f"C{format_length_mm(remote_c)}_D{format_length_mm(remote_d)}_"
                                            f"E{format_length_mm(remote_e)}_F{format_length_mm(remote_f)}_"
                                            f"G{format_length_mm(remote_g)}"
                                        ),
                                        "remote_a_motor_dx": remote_a,
                                        "remote_b_motor_dz": remote_b,
                                        "remote_c_top_link": remote_c,
                                        "remote_d_upper_link": remote_d,
                                        "remote_e_motor1_joint_offset": remote_e,
                                        "remote_f_drive_link": remote_f,
                                        "remote_g_distal_link": remote_g,
                                    })
        return cases

    raise ValueError(f"Unknown robot_type: {robot_type!r}")


def format_gain_value(value: float) -> str:
    return f"{value:.8g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class BatchCase:
    subject: str
    speed: float
    robot_type: str
    age_name: str
    height: float
    weight: float
    arm_name: str
    link1: float | None
    link2: float | None
    motor_distance: float | None
    remote_a_motor_dx: float | None
    remote_b_motor_dz: float | None
    remote_c_top_link: float | None
    remote_d_upper_link: float | None
    remote_e_motor1_joint_offset: float | None
    remote_f_drive_link: float | None
    remote_g_distal_link: float | None
    gain_name: str
    kp_motor1: float
    kd_motor1: float
    kp_motor2: float
    kd_motor2: float

    @property
    def outdir_name(self) -> str:
        if (
            np.isclose(self.kp_motor1, self.kp_motor2)
            and np.isclose(self.kd_motor1, self.kd_motor2)
        ):
            gain_suffix = f"Kp{int(self.kp_motor1)}"
        else:
            gain_suffix = (
                f"M1Kp{format_gain_value(self.kp_motor1)}Kd{format_gain_value(self.kd_motor1)}_"
                f"M2Kp{format_gain_value(self.kp_motor2)}Kd{format_gain_value(self.kd_motor2)}"
            )
        return (
            f"batch_{self.subject}_Spd{self.speed:.2f}_"
            f"{self.robot_type}_{self.age_name}_{self.arm_name}_{gain_suffix}"
        )

    @property
    def rel_outdir(self) -> str:
        return f"{save_dir}/{self.outdir_name}"

    @property
    def abs_outdir(self) -> Path:
        return ROOT / "outputs" / self.rel_outdir


SUMMARY_METRIC_COLUMNS = [
    "n_samples",
    "peak_tau",
    "rms_tau",
    "rms_tau1",
    "rms_tau2",
    "mean_abs_tau",
    "total_mech_work",
    "rmse_mm",
    "max_err_mm",
    "p95_err_mm",
    "tracking_fail_ratio",
]
SUMMARY_COLUMNS = [
    "case_index",
    *BatchCase.__dataclass_fields__.keys(),
    "kp",
    "kd",
    "outdir",
    "status",
    "returncode",
    "duration_s",
    "command",
    "error_tail",
    *SUMMARY_METRIC_COLUMNS,
]
# ─────────────────────────────────────────────────────────────────────────────
# 조건 생성
# ─────────────────────────────────────────────────────────────────────────────
def load_gait_cases(
    subject_filter: set[str] | None = None,
    speed_filter: set[float] | None = None,
) -> list[tuple[str, float]]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Cannot find gait data: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    if "Subject" not in df.columns or "Speed" not in df.columns:
        raise ValueError("gait_data.csv must contain 'Subject' and 'Speed' columns.")

    cases = (
        df[["Subject", "Speed"]]
        .drop_duplicates()
        .sort_values(["Subject", "Speed"])
        .itertuples(index=False, name=None)
    )

    out: list[tuple[str, float]] = []
    for subject, speed in cases:
        subject = str(subject)
        speed = float(speed)
        if subject_filter is not None and subject not in subject_filter:
            continue
        if speed_filter is not None and not any(
            np.isclose(speed, s, atol=SPEED_MATCH_ATOL) for s in speed_filter
        ):
            continue
        out.append((subject, speed))

    if not out:
        raise ValueError("No gait cases matched the requested filters.")
    return out


def make_cases(
    gait_cases: list[tuple[str, float]],
    robot_types: list[str],
    anthro_cases: list[dict],
    pd_gains: list[dict],
) -> list[BatchCase]:
    cases: list[BatchCase] = []
    for robot_type in robot_types:
        arm_cases = make_arm_cases(robot_type)
        for (subject, speed), anthro, arm, gains in itertools.product(
            gait_cases, anthro_cases, arm_cases, pd_gains
        ):
            kp_motor1 = float(gains.get("kp_motor1", gains.get("kp1", gains.get("kp"))))
            kd_motor1 = float(gains.get("kd_motor1", gains.get("kd1", gains.get("kd"))))
            kp_motor2 = float(gains.get("kp_motor2", gains.get("kp2", gains.get("kp", kp_motor1))))
            kd_motor2 = float(gains.get("kd_motor2", gains.get("kd2", gains.get("kd", kd_motor1))))
            cases.append(
                BatchCase(
                    subject=subject,
                    speed=float(speed),
                    robot_type=robot_type,
                    age_name=anthro["name"],
                    height=float(anthro["height"]),
                    weight=float(anthro["weight"]),
                    arm_name=arm["name"],
                    link1=float(arm["link1"]) if "link1" in arm else None,
                    link2=float(arm["link2"]) if "link2" in arm else None,
                    motor_distance=float(arm["motor_distance"]) if "motor_distance" in arm else None,
                    remote_a_motor_dx=float(arm["remote_a_motor_dx"]) if "remote_a_motor_dx" in arm else None,
                    remote_b_motor_dz=float(arm["remote_b_motor_dz"]) if "remote_b_motor_dz" in arm else None,
                    remote_c_top_link=float(arm["remote_c_top_link"]) if "remote_c_top_link" in arm else None,
                    remote_d_upper_link=float(arm["remote_d_upper_link"]) if "remote_d_upper_link" in arm else None,
                    remote_e_motor1_joint_offset=(
                        float(arm["remote_e_motor1_joint_offset"])
                        if "remote_e_motor1_joint_offset" in arm
                        else None
                    ),
                    remote_f_drive_link=float(arm["remote_f_drive_link"]) if "remote_f_drive_link" in arm else None,
                    remote_g_distal_link=float(arm["remote_g_distal_link"]) if "remote_g_distal_link" in arm else None,
                    gain_name=gains["name"],
                    kp_motor1=kp_motor1,
                    kd_motor1=kd_motor1,
                    kp_motor2=kp_motor2,
                    kd_motor2=kd_motor2,
                )
            )
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# 실행 및 결과 요약
# ─────────────────────────────────────────────────────────────────────────────
def build_command(case: BatchCase, render: str, cycles: int) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "src" / "simulate.py"),
        "--render", render,
        "--cycles", str(cycles),
        "--subject", case.subject,
        "--speed", f"{case.speed:.8g}",
        "--height", f"{case.height:.8g}",
        "--weight", f"{case.weight:.8g}",
        "--robot_type", case.robot_type,
        "--kp_motor1", f"{case.kp_motor1:.8g}",
        "--kd_motor1", f"{case.kd_motor1:.8g}",
        "--kp_motor2", f"{case.kp_motor2:.8g}",
        "--kd_motor2", f"{case.kd_motor2:.8g}",
    ]
    if case.link1 is not None:
        cmd.extend(["--link1", f"{case.link1:.8g}"])
    if case.link2 is not None:
        cmd.extend(["--link2", f"{case.link2:.8g}"])
    if case.motor_distance is not None:
        cmd.extend(["--remote_a_motor_dx", f"{case.motor_distance:.8g}"])
    if case.remote_a_motor_dx is not None:
        cmd.extend(["--remote_a_motor_dx", f"{case.remote_a_motor_dx:.8g}"])
    if case.remote_b_motor_dz is not None:
        cmd.extend(["--remote_b_motor_dz", f"{case.remote_b_motor_dz:.8g}"])
    if case.remote_c_top_link is not None:
        cmd.extend(["--remote_c_top_link", f"{case.remote_c_top_link:.8g}"])
    if case.remote_d_upper_link is not None:
        cmd.extend(["--remote_d_upper_link", f"{case.remote_d_upper_link:.8g}"])
    if case.remote_e_motor1_joint_offset is not None:
        cmd.extend(["--remote_e_motor1_joint_offset", f"{case.remote_e_motor1_joint_offset:.8g}"])
    if case.remote_f_drive_link is not None:
        cmd.extend(["--remote_f_drive_link", f"{case.remote_f_drive_link:.8g}"])
    if case.remote_g_distal_link is not None:
        cmd.extend(["--remote_g_distal_link", f"{case.remote_g_distal_link:.8g}"])
    cmd.extend(["--outdir", case.rel_outdir])
    return cmd


def summarize_torque_profile(csv_path: Path) -> dict[str, float | int | None]:
    """simulate.py가 저장한 torque_profile.csv에서 핵심 지표 계산."""
    if not csv_path.exists():
        return {
            "n_samples": 0,
            "peak_tau": None,
            "rms_tau": None,
            "rms_tau1": None,
            "rms_tau2": None,
            "mean_abs_tau": None,
            "total_mech_work": None,
            "rmse_mm": None,
            "max_err_mm": None,
            "p95_err_mm": None,
            "tracking_fail_ratio": None,
        }

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {
            "n_samples": 0,
            "peak_tau": None,
            "rms_tau": None,
            "rms_tau1": None,
            "rms_tau2": None,
            "mean_abs_tau": None,
            "total_mech_work": None,
            "rmse_mm": None,
            "max_err_mm": None,
            "p95_err_mm": None,
            "tracking_fail_ratio": None,
        }

    tau1 = df["motor1_torque"].to_numpy(float) if "motor1_torque" in df else np.zeros(len(df))
    tau2 = df["motor2_torque"].to_numpy(float) if "motor2_torque" in df else np.zeros(len(df))
    w1 = df["motor1_thetadot"].to_numpy(float) if "motor1_thetadot" in df else np.zeros(len(df))
    w2 = df["motor2_thetadot"].to_numpy(float) if "motor2_thetadot" in df else np.zeros(len(df))

    if "time" in df and len(df) > 1:
        dt = float(np.nanmedian(np.diff(df["time"].to_numpy(float))))
        if not np.isfinite(dt) or dt <= 0:
            dt = 0.001
    else:
        dt = 0.001

    rmse_mm = None
    max_err_mm = None
    p95_err_mm = None
    tracking_fail_ratio = None
    if {"ankle_actual_x", "ankle_actual_z", "ankle_target_x", "ankle_target_z"}.issubset(df.columns):
        err_x = df["ankle_actual_x"].to_numpy(float) - df["ankle_target_x"].to_numpy(float)
        err_z = df["ankle_actual_z"].to_numpy(float) - df["ankle_target_z"].to_numpy(float)
    elif {"ee_x", "ee_z", "ankle_target_x", "ankle_target_z"}.issubset(df.columns):
        err_x = df["ee_x"].to_numpy(float) - df["ankle_target_x"].to_numpy(float)
        err_z = df["ee_z"].to_numpy(float) - df["ankle_target_z"].to_numpy(float)
    else:
        err_x = None
        err_z = None

    if err_x is not None and err_z is not None:
        err_mm = np.sqrt(err_x**2 + err_z**2) * 1000.0
        rmse_mm = float(np.sqrt(np.mean(err_mm**2)))
        max_err_mm = float(np.max(err_mm))
        p95_err_mm = float(np.percentile(err_mm, 95))
        tracking_fail_ratio = float(np.mean(err_mm > 10.0))

    return {
        "n_samples": int(len(df)),
        "peak_tau": float(max(np.percentile(np.abs(tau1), 99), np.percentile(np.abs(tau2), 99))),
        "rms_tau": float(np.sqrt(np.mean(tau1**2 + tau2**2))),
        "rms_tau1": float(np.sqrt(np.mean(tau1**2))),
        "rms_tau2": float(np.sqrt(np.mean(tau2**2))),
        "mean_abs_tau": float((np.mean(np.abs(tau1)) + np.mean(np.abs(tau2))) / 2.0),
        "total_mech_work": float(np.sum(np.abs(tau1 * w1) + np.abs(tau2 * w2)) * dt),
        "rmse_mm": rmse_mm,
        "max_err_mm": max_err_mm,
        "p95_err_mm": p95_err_mm,
        "tracking_fail_ratio": tracking_fail_ratio,
    }


def run_one_case(
    case: BatchCase,
    render: str,
    cycles: int,
    skip_existing: bool,
    timeout: int | None,
) -> dict:
    case.abs_outdir.mkdir(parents=True, exist_ok=True)
    csv_path = case.abs_outdir / "torque_profile.csv"
    run_log = case.abs_outdir / "run.log"

    case_dict = asdict(case)
    case_dict["kp"] = case.kp_motor1
    case_dict["kd"] = case.kd_motor1

    row = {
        **case_dict,
        "outdir": str(case.abs_outdir.relative_to(ROOT)),
        "status": None,
        "returncode": None,
        "duration_s": None,
        "command": None,
        "error_tail": "",
    }

    if skip_existing and csv_path.exists():
        row.update({"status": "skipped_existing", "returncode": 0, "duration_s": 0.0})
        row.update(summarize_torque_profile(csv_path))
        return row

    cmd = build_command(case, render=render, cycles=cycles)
    row["command"] = " ".join(shlex.quote(c) for c in cmd)

    tic = time.time()
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    duration = time.time() - tic

    log_text = (
        f"$ {row['command']}\n\n"
        f"[STDOUT]\n{result.stdout}\n\n"
        f"[STDERR]\n{result.stderr}\n"
    )
    run_log.write_text(log_text, encoding="utf-8")

    status = "success"
    if result.returncode == 2:
        status = "failed_ik"
    elif result.returncode == 3:
        status = "failed_dynamic"
    elif result.returncode != 0:
        status = "failed_error"

    row.update(
        {
            "returncode": int(result.returncode),
            "duration_s": float(duration),
            "status": status,
            "error_tail": result.stderr[-1000:] if result.returncode != 0 else "",
        }
    )
    row.update(summarize_torque_profile(csv_path))

    # 작은 체형의 짧은 링크와 equality 구속 오차를 고려한 배치 판정값.
    # 30 mm는 소형 체형 전체 다리 길이(약 340 mm)의 9% 미만이다.
    if row["status"] == "success" and row.get("p95_err_mm") is not None and row["p95_err_mm"] > 30.0:
        row["status"] = "failed_tracking"

    if row["status"] != "success":
        if case.abs_outdir.exists():
            shutil.rmtree(case.abs_outdir, ignore_errors=True)

    return row


def load_recorded_summary_outdirs() -> set[str]:
    if not SUMMARY_CSV.exists() or SUMMARY_CSV.stat().st_size == 0:
        return set()

    recorded = set()
    try:
        with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("outdir"):
                    recorded.add(row["outdir"])
    except Exception as e:
        print(f"[WARN] 기존 summary를 읽지 못했습니다. 이번 실행에서는 summary skip을 적용하지 않습니다: {e}")
        return set()
    return recorded


def current_summary_columns() -> list[str]:
    if SUMMARY_CSV.exists() and SUMMARY_CSV.stat().st_size > 0:
        with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), None)
        if header:
            return header
    return SUMMARY_COLUMNS


def init_summary_outputs(preserve_summary: bool) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    if preserve_summary:
        SUMMARY_CSV.touch(exist_ok=True)
        FAILED_LOG.touch(exist_ok=True)
        return
    SUMMARY_CSV.write_text("", encoding="utf-8")
    FAILED_LOG.write_text("", encoding="utf-8")


def append_summary_row(row: dict) -> None:
    write_header = SUMMARY_CSV.stat().st_size == 0
    with SUMMARY_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=current_summary_columns(),
            extrasaction="ignore",
            restval="",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    status = str(row.get("status", ""))
    if not status.startswith("failed"):
        return

    with FAILED_LOG.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(str(row.get("outdir", "")) + "\n")
        f.write(
            "status={status}, rmse_mm={rmse}, max_err_mm={max_err}, "
            "p95_err_mm={p95}, tracking_fail_ratio={fail_ratio}\n".format(
                status=status,
                rmse=row.get("rmse_mm", ""),
                max_err=row.get("max_err_mm", ""),
                p95=row.get("p95_err_mm", ""),
                fail_ratio=row.get("tracking_fail_ratio", ""),
            )
        )
        f.write(str(row.get("command", "")) + "\n")
        f.write(str(row.get("error_tail", "")) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_csv_list(text: str | None, cast=str) -> set | None:
    if text is None or text.strip() == "":
        return None
    return {cast(x.strip()) for x in text.split(",") if x.strip()}


def make_error_row(case: BatchCase, status: str, duration_s: float | int | None, error_tail: str) -> dict:
    return {
        **asdict(case),
        "outdir": str(case.abs_outdir.relative_to(ROOT)),
        "status": status,
        "returncode": None,
        "duration_s": duration_s,
        "command": "",
        "error_tail": error_tail,
    }


def run_one_case_safe(
    case: BatchCase,
    render: str,
    cycles: int,
    skip_existing: bool,
    timeout: int | None,
) -> dict:
    try:
        return run_one_case(
            case,
            render=render,
            cycles=cycles,
            skip_existing=skip_existing,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        if case.abs_outdir.exists():
            shutil.rmtree(case.abs_outdir, ignore_errors=True)
        return make_error_row(case, "timeout", timeout, str(e))
    except Exception as e:
        if case.abs_outdir.exists():
            shutil.rmtree(case.abs_outdir, ignore_errors=True)
        return make_error_row(case, "exception", None, repr(e))


def status_label(status: str) -> str:
    if status == "success":
        return "✅"
    if status == "skipped_existing":
        return "↪ skipped"
    if status == "timeout":
        return "⏱️ TIMEOUT"
    if status == "exception":
        return "❌ EXCEPTION"
    if status == "failed_tracking":
        return "❌ tracking"
    return "❌"


def update_counts(row: dict, counts: dict[str, int]) -> None:
    status = row["status"]
    if status == "success":
        counts["success"] += 1
    elif status == "skipped_existing":
        counts["skipped"] += 1
    else:
        counts["failed"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="소아 보행 보조 로봇 대규모 배치 시뮬레이션")
    parser.add_argument("--render", choices=["none", "offscreen", "viewer"], default="none")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--max-runs", type=int, default=None, help="디버그용 최대 실행 개수")
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 조건 목록만 출력")
    parser.add_argument("--skip-existing", action="store_true", help="기존 torque_profile.csv가 있으면 건너뜀")
    parser.add_argument(
        "--append-summary",
        action="store_true",
        help="기존 batch_summary.csv와 실패 로그를 비우지 않고 새 결과를 뒤에 추가",
    )
    parser.add_argument("--timeout", type=int, default=None, help="각 run 제한 시간 [s]")
    parser.add_argument("--workers", type=int, default=1, help="동시에 실행할 batch 개수")
    parser.add_argument("--subjects", type=str, default=None, help="예: Sub01,Sub02")
    parser.add_argument(
        "--speeds",
        type=str,
        default=None,
        help="예: 0.45,0.912,1.146 (CSV 실제 Speed와 약간 달라도 매칭)",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default=",".join(ROBOT_TYPES),
        help="예: exoskeleton,open_chain",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  소아 보행 보조 로봇 대규모 배치 시뮬레이션")
    print("=" * 72)

    subject_filter = parse_csv_list(args.subjects, str)
    speed_filter = parse_csv_list(args.speeds, float)
    robot_types = sorted(parse_csv_list(args.robots, str) or set(ROBOT_TYPES), key=ROBOT_TYPES.index)

    gait_cases = load_gait_cases(subject_filter=subject_filter, speed_filter=speed_filter)
    cases = make_cases(
        gait_cases=gait_cases,
        robot_types=robot_types,
        anthro_cases=ANTHRO_CASES,
        pd_gains=PD_GAINS,
    )
    if args.max_runs is not None:
        cases = cases[: args.max_runs]

    print(f"보행 패턴: {len(gait_cases)}개")
    print(f"로봇 구조: {len(robot_types)}개 → {', '.join(robot_types)}")
    print(f"체형 조건: {len(ANTHRO_CASES)}개 → {', '.join(a['name'] for a in ANTHRO_CASES)}")
    arm_case_summary = ", ".join(f"{robot}:{len(make_arm_cases(robot))}" for robot in robot_types)
    print(f"기구 조건: {arm_case_summary}")
    print(f"PD gain 조건: {len(PD_GAINS)}개 → {', '.join(g['name'] for g in PD_GAINS)}")
    print(f"총 실행 조건: {len(cases)}개")
    print(f"render={args.render}, cycles={args.cycles}, skip_existing={args.skip_existing}, workers={args.workers}")
    print("=" * 72)

    if args.dry_run:
        for i, c in enumerate(cases, 1):
            print(f"[{i:04d}] {c.outdir_name}")
        print("\nDry run only. 실제 시뮬레이션은 실행하지 않았습니다.")
        return

    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    preserve_summary = (
        args.append_summary
        or (args.skip_existing and SUMMARY_CSV.exists() and SUMMARY_CSV.stat().st_size > 0)
    )
    skip_recorded_summary = args.skip_existing and not args.append_summary
    recorded_summary_outdirs = load_recorded_summary_outdirs() if skip_recorded_summary else set()
    init_summary_outputs(preserve_summary=preserve_summary)

    indexed_cases = list(enumerate(cases, 1))
    if recorded_summary_outdirs:
        pending_cases = [
            (i, case)
            for i, case in indexed_cases
            if str(case.abs_outdir.relative_to(ROOT)) not in recorded_summary_outdirs
        ]
    else:
        pending_cases = indexed_cases

    summary_skipped = len(indexed_cases) - len(pending_cases)
    if summary_skipped > 0:
        print(f"기존 batch_summary.csv 기록: {summary_skipped}개 건너뜀")

    counts = {"success": 0, "failed": 0, "skipped": summary_skipped}
    total_completed = summary_skipped

    if args.workers <= 1:
        for i, case in pending_cases:
            print(f"[{i}/{len(cases)}] {case.outdir_name} ... ", end="", flush=True)
            row = run_one_case_safe(
                case,
                render=args.render,
                cycles=args.cycles,
                skip_existing=args.skip_existing,
                timeout=args.timeout,
            )
            row["case_index"] = i
            total_completed += 1
            update_counts(row, counts)
            print(status_label(row["status"]))

            # 중간에 중단되어도 결과를 최대한 보존
            append_summary_row(row)
    else:
        worker_count = max(1, min(args.workers, len(pending_cases)))
        next_case_idx = 0
        completed = 0
        pending = {}

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            while next_case_idx < len(pending_cases) and len(pending) < worker_count:
                i, case = pending_cases[next_case_idx]
                future = executor.submit(
                    run_one_case_safe,
                    case,
                    args.render,
                    args.cycles,
                    args.skip_existing,
                    args.timeout,
                )
                pending[future] = (i, case)
                print(f"[queued case {i}/{len(cases)}] {case.outdir_name}")
                next_case_idx += 1

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i, case = pending.pop(future)
                    row = future.result()
                    row["case_index"] = i
                    completed += 1
                    total_completed += 1
                    update_counts(row, counts)
                    print(
                        f"[done {total_completed}/{len(cases)} | case {i}] "
                        f"{case.outdir_name} ... {status_label(row['status'])}",
                        flush=True,
                    )

                    # 중간에 중단되어도 결과를 최대한 보존
                    append_summary_row(row)

                    if next_case_idx < len(pending_cases):
                        next_i, next_case = pending_cases[next_case_idx]
                        next_future = executor.submit(
                            run_one_case_safe,
                            next_case,
                            args.render,
                            args.cycles,
                            args.skip_existing,
                            args.timeout,
                        )
                        pending[next_future] = (next_i, next_case)
                        print(f"[queued case {next_i}/{len(cases)}] {next_case.outdir_name}")
                        next_case_idx += 1

    print("\n" + "=" * 72)
    print("배치 작업 완료")
    print(f"성공: {counts['success']}, 실패: {counts['failed']}, 스킵: {counts['skipped']}, 총: {total_completed}")
    print(f"요약 CSV: {SUMMARY_CSV}")
    if counts["failed"] > 0:
        print(f"실패 로그: {FAILED_LOG}")
    print(f"개별 결과: {BATCH_ROOT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
