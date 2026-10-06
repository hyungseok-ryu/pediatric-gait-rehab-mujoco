"""
simulate.py
===========
소아 보행 보조 외골격/원격 평행사변형 MuJoCo 시뮬레이션 메인 루프.

실행:
    python src/simulate.py [--render {viewer|offscreen|none}] [--cycles N]

파이프라인 (project.md §5.1):
  1. 환자 파라미터 로드 + 인체측정학 추정 (anthropometry)
  2. 보행 데이터 로드 + 보간 (gait_loader)
  3. 사람 ankle 위치 시계열 계산 (kinematics)
  4. MuJoCo XML 동적 생성 (build_xml)
  5. MuJoCo 시뮬레이션 루프:
       - 사람 다리 position actuator → kinematic playback
       - 외골격: hip/knee 목표각 직접 추종
       - remote_parallelogram: IK → 목표 모터각
       - PD 제어 → 모터 토크 적용
  6. 결과 CSV 저장
  7. 결과 후처리
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

# 프로젝트 내부 모듈 (src/ 에서 실행 시 상위 경로 추가)
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config as cfg
from anthropometry import estimate_open_chain_link_lengths, estimate_segment_params
from gait_loader import load_gait_data
from kinematics import (
    compute_ankle_trajectory,
    open_chain_ik,
    remote_ankle_arm_ik,
    remote_exoskeleton_ik,
    robot_arm_ik,
)
from controller import build_pd_controllers
from build_xml import build_xml
try:
    from plot_results import plot_all
except ModuleNotFoundError:
    plot_all = None


def compute_target_ee(target_ankle, base_pos, cuff_length):
    """
    브라켓 끝이 target_ankle에 정확히 위치하도록 로봇팔의 목표 EE 좌표를 기하학적으로 계산합니다.
    """
    Yb = base_pos[1]
    Zb = base_pos[2]
    Za = target_ankle[2]
    
    D_yz = np.sqrt(Yb**2 + (Za - Zb)**2)
    L_yz = np.sqrt(max(0, D_yz**2 - cuff_length**2))
        
    phi = np.arctan2(-Yb, -(Za - Zb))
    alpha = np.arctan2(cuff_length, L_yz)
    theta_p = phi + alpha
    
    ee_x = target_ankle[0]
    ee_y = cuff_length * np.cos(theta_p)
    ee_z = Za + cuff_length * np.sin(theta_p)
    
    return np.array([ee_x, ee_y, ee_z])
# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """1차 유한 차분으로 속도 시계열 계산."""
    vel = np.gradient(arr, dt)
    return vel


def _quintic_profile(elapsed: float, duration: float) -> tuple[float, float]:
    """양 끝에서 속도와 가속도가 0인 진행률과 진행률 속도를 반환한다."""
    duration = max(float(duration), np.finfo(float).eps)
    u = float(np.clip(elapsed / duration, 0.0, 1.0))
    position = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    return position, velocity


def _contralateral_trajectory(
    values: np.ndarray,
    cycle_duration: float,
    dt: float,
) -> np.ndarray:
    """오른쪽 주기 파형을 50% 앞당겨 왼쪽 보행 파형을 만든다."""
    half_cycle_steps = int(round(0.5 * cycle_duration / dt))
    if half_cycle_steps <= 0:
        raise ValueError("Gait cycle is too short for a bilateral phase shift")
    return np.roll(np.asarray(values, dtype=float), -half_cycle_steps)


def _get_sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """센서 이름으로 sensordata 슬라이스를 반환한다."""
    sid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = model.sensor_adr[sid]
    dim   = model.sensor_dim[sid]
    return data.sensordata[start:start + dim]


def _get_actuator_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _get_joint_qpos_id(model: mujoco.MjModel, name: str) -> int:
    """관절 이름 → qpos 인덱스."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]


def _get_joint_qvel_id(model: mujoco.MjModel, name: str) -> int:
    """관절 이름 → qvel 인덱스."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_dofadr[jid]


def _remote_geometry(arm) -> dict[str, float]:
    """사용자 스케치의 remote_* 파라미터를 IK/XML 인자로 묶는다."""
    return {
        "link1_length": getattr(arm, "remote_f_drive_link", arm.link1_length),
        "link2_length": getattr(arm, "remote_g_distal_link", arm.link2_length),
        "motor_distance": getattr(arm, "remote_a_motor_dx", arm.motor_distance),
        "motor2_offset_z": getattr(arm, "remote_b_motor_dz", arm.motor2_offset_z),
        "remote_c_top_link": arm.remote_c_top_link,
        "remote_d_upper_link": arm.remote_d_upper_link,
        "remote_e_motor1_joint_offset": arm.remote_e_motor1_joint_offset,
    }


def _apply_joint_range_limits(
    xml_text: str,
    robot_type: str,
    ranges_deg: dict[str, tuple[float, float]],
) -> str:
    """UI에서 설정한 좌우 hip/knee 범위를 MuJoCo hard limit에 반영한다."""
    root = ET.fromstring(xml_text)
    joint_ranges = {
        "hip_joint": ranges_deg["right_hip"],
        "knee_joint": ranges_deg["right_knee"],
        "hip_joint_left": ranges_deg["left_hip"],
        "knee_joint_left": ranges_deg["left_knee"],
    }
    if robot_type == "exoskeleton":
        joint_ranges.update(
            {
                "motor1_joint": ranges_deg["right_hip"],
                "motor2_joint": ranges_deg["right_knee"],
                "motor1_joint_left": ranges_deg["left_hip"],
                "motor2_joint_left": ranges_deg["left_knee"],
            }
        )

    found: set[str] = set()
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        if name not in joint_ranges:
            continue
        minimum, maximum = joint_ranges[name]
        joint.attrib["limited"] = "true"
        joint.attrib["range"] = (
            f"{np.deg2rad(minimum):.10g} {np.deg2rad(maximum):.10g}"
        )
        found.add(name)

    missing = set(joint_ranges) - found
    if missing:
        raise ValueError(f"관절 범위를 적용할 joint를 찾지 못했습니다: {sorted(missing)}")

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root,
        encoding="unicode",
    ) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# 메인 시뮬레이션
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    render_mode: str = "viewer", 
    n_cycles: int = 3,
    subject_id: str | None = None,
    speed: float | None = None,
    height: float | None = None,
    weight: float | None = None,
    hip_joint_spacing: float | None = None,
    robot_type: str | None = None,
    kp_val: float | None = None,
    kd_val: float | None = None,
    kp_motor1: float | None = None,
    kd_motor1: float | None = None,
    kp_motor2: float | None = None,
    kd_motor2: float | None = None,
    link1: float | None = None,
    link2: float | None = None,
    motor2_dz: float | None = None,
    remote_a_motor_dx: float | None = None,
    remote_b_motor_dz: float | None = None,
    remote_c_top_link: float | None = None,
    remote_d_upper_link: float | None = None,
    remote_e_motor1_joint_offset: float | None = None,
    remote_f_drive_link: float | None = None,
    remote_g_distal_link: float | None = None,
    out_dir_name: str | None = None,
    save_video: bool | None = None,
    video_path: str | Path | None = None,
    camera_name: str = "fixed",
    gait_data_path: str | Path | None = None,
    cycle_duration: float | None = None,
    right_hip_range_deg: tuple[float, float] | None = None,
    right_knee_range_deg: tuple[float, float] | None = None,
    left_hip_range_deg: tuple[float, float] | None = None,
    left_knee_range_deg: tuple[float, float] | None = None,
    runtime_control=None,
) -> None:
    """
    전체 시뮬레이션 파이프라인을 실행한다.

    Parameters
    ----------
    render_mode : "viewer" | "offscreen" | "none"
    n_cycles    : 보행 사이클 반복 횟수
    save_video  : True이면 render_mode와 별개로 mp4를 저장. None이면 offscreen에서만 저장.
    camera_name : "fixed"는 운동학용 시상면, "presentation"은 actuator 확인용 사선 시점.
    gait_data_path : 생략하면 config의 기본 3–4세 Comfortable 평균 궤적 사용.
    cycle_duration : 지정하면 입력 각도 파형의 1주기 시간을 해당 값[s]으로 조정.
    *_range_deg : UI 세션에서 사용하는 좌우 hip/knee 최소·최대 각도[deg].
        지정하면 목표 궤적과 MuJoCo joint hard limit에 함께 적용한다.
    runtime_control : UI에서 전달하는 multiprocessing shared mapping. None이면 기존
        배치 시뮬레이션과 동일하게 동작한다.
    """
    interactive = runtime_control is not None
    if interactive:
        try:
            runtime_control["status"] = "initializing"
            runtime_control["actual_state"] = "initializing"
        except (BrokenPipeError, ConnectionError, EOFError):
            runtime_control = None
            interactive = False
    # ── 인자 기반 설정 덮어쓰기 (Overrides) ─────────────────────────
    if height is not None: cfg.SUBJECT.subject_height = height
    if weight is not None: cfg.SUBJECT.subject_weight = weight
    if hip_joint_spacing is not None:
        cfg.SUBJECT.hip_joint_spacing = hip_joint_spacing
    if robot_type is not None: cfg.ROBOT_ARM.robot_type = robot_type
    supported_robot_types = {
        "exoskeleton",
        "open_chain",
    }
    if cfg.ROBOT_ARM.robot_type not in supported_robot_types:
        raise ValueError(
            f"지원하지 않는 robot_type={cfg.ROBOT_ARM.robot_type!r}. "
            f"가능한 값: {', '.join(sorted(supported_robot_types))}"
        )
    if camera_name not in {"fixed", "presentation"}:
        raise ValueError("camera_name must be 'fixed' or 'presentation'")

    range_inputs = {
        "right_hip": right_hip_range_deg,
        "right_knee": right_knee_range_deg,
        "left_hip": left_hip_range_deg,
        "left_knee": left_knee_range_deg,
    }
    range_limit_enabled = any(value is not None for value in range_inputs.values())
    if range_limit_enabled and not all(
        value is not None for value in range_inputs.values()
    ):
        raise ValueError("관절 범위는 좌우 hip/knee 네 항목을 모두 지정해야 합니다.")
    joint_ranges_deg: dict[str, tuple[float, float]] = {}
    if range_limit_enabled:
        for name, value in range_inputs.items():
            assert value is not None
            minimum, maximum = map(float, value)
            if not np.isfinite([minimum, maximum]).all() or minimum >= maximum:
                raise ValueError(f"잘못된 관절 범위 {name}: {value}")
            joint_ranges_deg[name] = (minimum, maximum)
        if not (
            joint_ranges_deg["right_hip"][0]
            <= 0.0
            <= joint_ranges_deg["right_hip"][1]
            and joint_ranges_deg["left_hip"][0]
            <= 0.0
            <= joint_ranges_deg["left_hip"][1]
            and joint_ranges_deg["right_knee"][0]
            <= 5.0
            <= joint_ranges_deg["right_knee"][1]
            and joint_ranges_deg["left_knee"][0]
            <= 5.0
            <= joint_ranges_deg["left_knee"][1]
        ):
            raise ValueError(
                "관절 범위에 중립 기립자세(hip 0°, knee 5°)가 포함되어야 합니다."
            )
    if kp_val is not None: 
        cfg.PD_GAINS.Kp_motor1 = kp_val
        cfg.PD_GAINS.Kp_motor2 = kp_val
    if kd_val is not None:
        cfg.PD_GAINS.Kd_motor1 = kd_val
        cfg.PD_GAINS.Kd_motor2 = kd_val
    if kp_motor1 is not None: cfg.PD_GAINS.Kp_motor1 = kp_motor1
    if kd_motor1 is not None: cfg.PD_GAINS.Kd_motor1 = kd_motor1
    if kp_motor2 is not None: cfg.PD_GAINS.Kp_motor2 = kp_motor2
    if kd_motor2 is not None: cfg.PD_GAINS.Kd_motor2 = kd_motor2
    if link1 is not None:
        cfg.ROBOT_ARM.link1_length = link1
        cfg.ROBOT_ARM.open_chain_link1_length = link1
        if hasattr(cfg.ROBOT_ARM, "remote_ankle_link1_length"):
            cfg.ROBOT_ARM.remote_ankle_link1_length = link1
        if hasattr(cfg.ROBOT_ARM, "remote_f_drive_link"):
            cfg.ROBOT_ARM.remote_f_drive_link = link1
    if link2 is not None:
        cfg.ROBOT_ARM.link2_length = link2
        cfg.ROBOT_ARM.open_chain_link2_length = link2
        if hasattr(cfg.ROBOT_ARM, "remote_ankle_link2_length"):
            cfg.ROBOT_ARM.remote_ankle_link2_length = link2
        if hasattr(cfg.ROBOT_ARM, "remote_g_distal_link"):
            cfg.ROBOT_ARM.remote_g_distal_link = link2
    if motor2_dz is not None:
        cfg.ROBOT_ARM.motor2_offset_z = motor2_dz
        if hasattr(cfg.ROBOT_ARM, "remote_b_motor_dz"):
            cfg.ROBOT_ARM.remote_b_motor_dz = motor2_dz
    if remote_a_motor_dx is not None:
        cfg.ROBOT_ARM.remote_a_motor_dx = remote_a_motor_dx
        cfg.ROBOT_ARM.motor_distance = remote_a_motor_dx
    if remote_b_motor_dz is not None:
        cfg.ROBOT_ARM.remote_b_motor_dz = remote_b_motor_dz
        cfg.ROBOT_ARM.motor2_offset_z = remote_b_motor_dz
    if remote_c_top_link is not None:
        cfg.ROBOT_ARM.remote_c_top_link = remote_c_top_link
    if remote_d_upper_link is not None:
        cfg.ROBOT_ARM.remote_d_upper_link = remote_d_upper_link
    if remote_e_motor1_joint_offset is not None:
        cfg.ROBOT_ARM.remote_e_motor1_joint_offset = remote_e_motor1_joint_offset
    if remote_f_drive_link is not None:
        cfg.ROBOT_ARM.remote_f_drive_link = remote_f_drive_link
        cfg.ROBOT_ARM.link1_length = remote_f_drive_link
    if remote_g_distal_link is not None:
        cfg.ROBOT_ARM.remote_g_distal_link = remote_g_distal_link
        cfg.ROBOT_ARM.link2_length = remote_g_distal_link
    print("=" * 60)
    print("  소아 보행 보조 로봇 MuJoCo 시뮬레이션")
    print("=" * 60)

    # ── Step 1: 환자 파라미터 / 인체측정학 ────────────────────────────────
    subj = cfg.SUBJECT
    seg = estimate_segment_params(
        height=subj.subject_height,
        weight=subj.subject_weight,
        thigh_length=subj.thigh_length,
        shank_length=subj.shank_length,
        thigh_mass=subj.thigh_mass,
        shank_mass=subj.shank_mass,
        foot_mass=subj.foot_mass,
    )
    print(f"\n[1] 대상자 파라미터")
    print(f"    키={subj.subject_height:.2f}m, 몸무게={subj.subject_weight:.1f}kg")
    print(f"    대퇴 길이={seg.thigh_length:.3f}m, 하퇴 길이={seg.shank_length:.3f}m")
    print(f"    대퇴 질량={seg.thigh_mass:.3f}kg, 하퇴 질량={seg.shank_mass:.3f}kg")
    hip_spacing = (
        subj.hip_joint_spacing
        if subj.hip_joint_spacing is not None
        else cfg.HIP_JOINT_SPACING_HEIGHT_RATIO * subj.subject_height
    )
    if hip_spacing <= 0:
        raise ValueError("hip_joint_spacing must be positive")
    print(
        f"    좌우 hip joint 간격={hip_spacing:.3f}m "
        f"({'입력값' if subj.hip_joint_spacing is not None else '키의 16% 추정'})"
    )
    walker = cfg.WALKER
    walker_width = (
        walker.frame_width
        if walker.frame_width is not None
        else float(np.clip(0.68 * subj.subject_height, 0.54, 0.66))
    )
    walker_width = max(
        walker_width,
        2.0 * (0.5 * hip_spacing + abs(cfg.ROBOT_ARM.exo_lateral_offset) + 0.10),
    )
    walker_length = (
        walker.frame_length
        if walker.frame_length is not None
        else float(np.clip(0.75 * subj.subject_height, 0.62, 0.78))
    )
    if walker.enabled:
        print(
            f"    고정식 수동보행기={walker_width:.3f}m(W) × "
            f"{walker_length:.3f}m(L), 바퀴는 시각화 전용"
        )

    # 로봇 link 길이 계산
    arm = cfg.ROBOT_ARM
    is_open_chain = arm.robot_type == "open_chain"
    is_remote_ee = arm.robot_type == "remote_parallelogram"
    is_remote_ankle = arm.robot_type == "remote_ankle_arm"
    is_remote_exo = arm.robot_type == "remote_exoskeleton"
    remote_geom = _remote_geometry(arm) if is_remote_ee else {}
    if arm.robot_type in {"exoskeleton", "remote_exoskeleton"}:
        link1_len = seg.thigh_length
        link2_len = seg.shank_length
    elif is_open_chain:
        auto_link1, auto_link2 = estimate_open_chain_link_lengths(
            seg.thigh_length,
            seg.shank_length,
            arm.open_chain_reach_margin,
        )
        link1_len = (
            auto_link1
            if arm.open_chain_auto_size and link1 is None
            else arm.open_chain_link1_length
        )
        link2_len = (
            auto_link2
            if arm.open_chain_auto_size and link2 is None
            else arm.open_chain_link2_length
        )
    elif is_remote_ankle:
        link1_len = arm.remote_ankle_link1_length
        link2_len = arm.remote_ankle_link2_length
    else:
        link1_len = remote_geom["link1_length"]
        link2_len = remote_geom["link2_length"]
    ik_motor_distance = (
        arm.remote_exo_motor_dx
        if is_remote_ankle
        else remote_geom.get("motor_distance", arm.motor_distance)
    )
    ik_motor2_offset_z = (
        arm.remote_exo_motor_dz
        if is_remote_ankle
        else remote_geom.get("motor2_offset_z", arm.motor2_offset_z)
    )
    ik_remote_c_top_link = remote_geom.get("remote_c_top_link", arm.remote_c_top_link)
    ik_remote_d_upper_link = remote_geom.get("remote_d_upper_link", arm.remote_d_upper_link)
    ik_remote_e_motor1_joint_offset = remote_geom.get(
        "remote_e_motor1_joint_offset",
        arm.remote_e_motor1_joint_offset,
    )
    print(f"\n    {arm.robot_type} link1={link1_len:.3f}m, link2={link2_len:.3f}m")
    if is_open_chain:
        uses_auto_values = (
            arm.open_chain_auto_size
            and np.isclose(link1_len, auto_link1)
            and np.isclose(link2_len, auto_link2)
        )
        sizing_mode = "체형 자동 산정값" if uses_auto_values else "수동/부분 override"
        print(
            f"    open-chain 링크 설정={sizing_mode}, "
            f"도달 여유율={arm.open_chain_reach_margin * 100:.0f}%"
        )
    if arm.robot_type in {
        "exoskeleton",
        "open_chain",
        "remote_ankle_arm",
        "remote_exoskeleton",
    }:
        base_moving_mass = 2.0 * (arm.exo_motor_mass + arm.exo_link_mass)
        link_mass_label = (
            "link1/link2 구조 링크"
            if is_open_chain or is_remote_ankle
            else "thigh/shank 구조 링크"
        )
        print(
            f"    actuator={arm.exo_motor_mass:.3f}kg × 2, "
            f"{link_mass_label}={arm.exo_link_mass:.3f}kg × 2"
        )
        if is_remote_ankle or is_remote_exo:
            transmission_mass = (
                arm.remote_exo_plate_mass
                + arm.remote_exo_coupler_mass
                + arm.remote_exo_crank_mass
                + arm.remote_exo_rod_mass
            )
            print(
                "    전달부 plate/coupler/crank/rod="
                f"{arm.remote_exo_plate_mass:.3f}/"
                f"{arm.remote_exo_coupler_mass:.3f}/"
                f"{arm.remote_exo_crank_mass:.3f}/"
                f"{arm.remote_exo_rod_mass:.3f}kg"
            )
            base_moving_mass += transmission_mass
        print(
            f"    외골격 moving assembly 질량="
            f"{base_moving_mass:.3f}kg/측, {2.0 * base_moving_mass:.3f}kg/양측"
        )

    # ── Step 2: 보행 데이터 로드 ──────────────────────────────────────────
    gait_csv = Path(gait_data_path) if gait_data_path is not None else cfg.PATHS["gait_data_csv"]
    effective_cycle_duration = cycle_duration
    if effective_cycle_duration is None and gait_data_path is None:
        effective_cycle_duration = cfg.DEFAULT_GAIT_CYCLE_DURATION
    print(f"\n[2] 보행 데이터 로드: {gait_csv}")
    # 실시간 UI는 동일한 한 주기 파형을 순환 재생하므로 전체 세션 길이만큼
    # 궤적/IK 배열을 복제하지 않는다. n_cycles는 UI 세션의 최대 실행시간을
    # 결정하는 용도로 유지한다.
    trajectory_cycles = 1 if interactive else n_cycles
    traj = load_gait_data(
        csv_path=str(gait_csv),
        col_map=cfg.CSV_COLUMN_MAP,
        dt_csv=cfg.GAIT_DATA_DT,
        target_dt=cfg.SIM_TIMESTEP,
        n_cycles=trajectory_cycles,
        subject_id=subject_id,
        speed=speed,
        cycle_duration_override=effective_cycle_duration,
    )
    N = len(traj.time)
    simulation_steps = (
        max(1, int(round(n_cycles * traj.cycle_duration / cfg.SIM_TIMESTEP)))
        if interactive
        else N
    )
    print(
        f"    시간 스케일: 원본 {traj.source_cycle_duration:.4f}s → "
        f"시뮬레이션 {traj.cycle_duration:.4f}s "
        f"({traj.temporal_scale:.3f}×)"
    )
    print(f"    총 {N} 스텝, 기간={traj.time[-1]:.2f}s")

    # ── Step 3: ankle 위치 시계열 계산 ───────────────────────────────────
    # 입력 파일은 오른쪽 한 주기이므로 왼쪽은 정상 교대보행 가정에 따라
    # 제한 전 원본 파형을 50% cycle 앞당겨 생성한다. 이후 각 측의 환자별
    # ROM을 독립적으로 적용한다.
    source_right_hip_angle = traj.hip_angle.copy()
    source_right_knee_angle = traj.knee_angle.copy()
    left_hip_angle = _contralateral_trajectory(
        source_right_hip_angle,
        traj.cycle_duration,
        cfg.SIM_TIMESTEP,
    )
    left_knee_angle = _contralateral_trajectory(
        source_right_knee_angle,
        traj.cycle_duration,
        cfg.SIM_TIMESTEP,
    )
    if range_limit_enabled:
        right_hip_min, right_hip_max = np.deg2rad(
            joint_ranges_deg["right_hip"]
        )
        right_knee_min, right_knee_max = np.deg2rad(
            joint_ranges_deg["right_knee"]
        )
        left_hip_min, left_hip_max = np.deg2rad(
            joint_ranges_deg["left_hip"]
        )
        left_knee_min, left_knee_max = np.deg2rad(
            joint_ranges_deg["left_knee"]
        )
        traj.hip_angle = np.clip(
            source_right_hip_angle,
            right_hip_min,
            right_hip_max,
        )
        traj.knee_angle = np.clip(
            source_right_knee_angle,
            right_knee_min,
            right_knee_max,
        )
        left_hip_angle = np.clip(left_hip_angle, left_hip_min, left_hip_max)
        left_knee_angle = np.clip(
            left_knee_angle,
            left_knee_min,
            left_knee_max,
        )
        print(
            "    관절 구동 범위[deg]: "
            f"오른쪽 hip {joint_ranges_deg['right_hip']}, "
            f"knee {joint_ranges_deg['right_knee']} / "
            f"왼쪽 hip {joint_ranges_deg['left_hip']}, "
            f"knee {joint_ranges_deg['left_knee']}"
        )
    right_hip_global = np.array([0.0, -0.5 * hip_spacing, 0.0])
    left_hip_global = np.array([0.0, 0.5 * hip_spacing, 0.0])

    # 스케일 팩터: 대상자 다리 길이 / 보행 데이터 기준 다리 길이 (동일 subject이면 1.0)
    # Assumption: CSV 데이터는 동일 대상자 또는 스케일 on/off 옵션
    scale = 1.0
    if subj.scale_trajectory:
        # 기준 길이는 Winter 비율로 추정된 것 대비 실제 입력값 비율
        # (이미 동일 값이면 scale=1.0)
        default_thigh = 0.245 * subj.subject_height
        scale = seg.thigh_length / default_thigh if default_thigh > 0 else 1.0

    ankle_traj = compute_ankle_trajectory(
        hip_angles=traj.hip_angle,
        knee_angles=traj.knee_angle,
        thigh_length=seg.thigh_length,
        shank_length=seg.shank_length,
        hip_pos=right_hip_global,
        scale_factor=scale,
    )
    left_ankle_traj = compute_ankle_trajectory(
        hip_angles=left_hip_angle,
        knee_angles=left_knee_angle,
        thigh_length=seg.thigh_length,
        shank_length=seg.shank_length,
        hip_pos=left_hip_global,
        scale_factor=scale,
    )

    # UI의 기립 유지 기준자세. 양쪽을 같은 자세로 두어 보행 파형의 첫
    # 샘플(좌우 비대칭 보행자세)이 정지자세로 사용되는 것을 막는다.
    stand_hip_angle = 0.0
    stand_knee_angle = np.deg2rad(5.0)
    stand_ankle = compute_ankle_trajectory(
        hip_angles=np.array([stand_hip_angle]),
        knee_angles=np.array([stand_knee_angle]),
        thigh_length=seg.thigh_length,
        shank_length=seg.shank_length,
        hip_pos=right_hip_global,
        scale_factor=scale,
    )[0]
    left_stand_ankle = compute_ankle_trajectory(
        hip_angles=np.array([stand_hip_angle]),
        knee_angles=np.array([stand_knee_angle]),
        thigh_length=seg.thigh_length,
        shank_length=seg.shank_length,
        hip_pos=left_hip_global,
        scale_factor=scale,
    )[0]

    # Sit-to-Stand의 앉은 자세는 임상 궤적이 제공되지 않은 현재 연구용
    # 모델의 명시적 가정이다. 이상값(hip 45°, knee 75°)을 환자별 ROM으로
    # 제한하여 좌우 각각 적용한다. 골반이 고정된 모델이므로 실제 체중부하
    # 일어서기가 아니라 hip/knee 전환 궤적의 시각·제어 검증에 해당한다.
    def bounded_sit_angle(name: str, ideal_deg: float) -> float:
        if range_limit_enabled:
            minimum, maximum = joint_ranges_deg[name]
            ideal_deg = float(np.clip(ideal_deg, minimum, maximum))
        return float(np.deg2rad(ideal_deg))

    right_sit_pose = np.array(
        [
            bounded_sit_angle("right_hip", 45.0),
            bounded_sit_angle("right_knee", 75.0),
        ],
        dtype=float,
    )
    left_sit_pose = np.array(
        [
            bounded_sit_angle("left_hip", 45.0),
            bounded_sit_angle("left_knee", 75.0),
        ],
        dtype=float,
    )

    # 목표 각속도 (유한 차분)
    hip_vel_des  = _finite_diff(traj.hip_angle,  cfg.SIM_TIMESTEP)
    knee_vel_des = _finite_diff(traj.knee_angle, cfg.SIM_TIMESTEP)
    left_hip_vel_des = _finite_diff(left_hip_angle, cfg.SIM_TIMESTEP)
    left_knee_vel_des = _finite_diff(left_knee_angle, cfg.SIM_TIMESTEP)

    # ── Step 4: MuJoCo XML 생성 + 모델 로드 ──────────────────────────────
    print("\n[3] MuJoCo XML 생성 및 모델 로드")

    if arm.robot_type in {
        "exoskeleton",
        "open_chain",
        "remote_ankle_arm",
        "remote_exoskeleton",
    }:
        # build_xml은 이 중심선 기준 오른쪽 위치를 받은 뒤, 양측화 단계에서
        # 다리와 함께 ±hip_spacing/2만큼 이동한다.
        base_pos = (0.0, -abs(arm.exo_lateral_offset), 0.0)
        right_base_pos = np.array(
            [base_pos[0], base_pos[1] - 0.5 * hip_spacing, base_pos[2]]
        )
        left_base_pos = np.array(
            [base_pos[0], -base_pos[1] + 0.5 * hip_spacing, base_pos[2]]
        )
        print(
            f"    {arm.robot_type} base Y: 오른쪽={right_base_pos[1]:.3f}m, "
            f"왼쪽={left_base_pos[1]:.3f}m"
        )
    else:
        # remote_parallelogram base: 무릎 높이보다 base_above_knee만큼 위
        knee_z = right_hip_global[2] - seg.thigh_length
        base_pos_z = knee_z + arm.base_above_knee
        base_pos = (arm.x_offset, arm.y_offset, base_pos_z)
        print(
            f"    원격 평행사변형 base 위치: x={base_pos[0]:.3f}, "
            f"y={base_pos[1]:.3f}, z={base_pos[2]:.3f}"
        )

    # ── Step 3.5: IK 도달 가능성 사전 검증 (Pre-check) ──────────────────────────
    print("\n[3.5] 역기구학 도달 가능성 사전 검증")
    base_pos_arr = right_base_pos if arm.robot_type in {
        "exoskeleton",
        "open_chain",
        "remote_ankle_arm",
        "remote_exoskeleton",
    } else np.array(base_pos)
    arm_ik_kwargs = dict(
        d_motor=arm.motor_offset_z,
        robot_type=arm.robot_type,
        motor_distance=ik_motor_distance,
        motor2_offset_z=ik_motor2_offset_z,
        remote_c_top_link=ik_remote_c_top_link,
        remote_d_upper_link=ik_remote_d_upper_link,
        remote_e_motor1_joint_offset=ik_remote_e_motor1_joint_offset,
        distal_standoff=getattr(arm, "distal_standoff", 0.020),
    )
    open_chain_targets = None
    open_chain_motor_vel = None
    left_open_chain_targets = None
    left_open_chain_motor_vel = None
    stand_robot_targets = np.array([stand_hip_angle, stand_knee_angle])
    left_stand_robot_targets = stand_robot_targets.copy()
    remote_ankle_targets = None
    remote_ankle_motor_vel = None
    remote_exo_targets = None
    remote_exo_motor_vel = None
    if is_open_chain:
        open_chain_results = [
            open_chain_ik(
                target_pos=target_ankle,
                base_pos=base_pos_arr,
                link1_length=link1_len,
                link2_length=link2_len,
                elbow_behind=arm.open_chain_elbow_behind,
            )
            for target_ankle in ankle_traj
        ]
        pre_ik_fail_count = sum(
            result is None for result in open_chain_results
        )
        if pre_ik_fail_count:
            print(
                f"    [FAIL] open_chain IK 실패: "
                f"{pre_ik_fail_count}/{N} 스텝"
            )
            print("    open_chain link1/link2 길이를 확인하세요.")
            sys.exit(2)

        open_chain_targets = np.asarray(open_chain_results, dtype=float)
        open_chain_targets[:, 0] = np.unwrap(open_chain_targets[:, 0])
        open_chain_targets[:, 1] = np.unwrap(open_chain_targets[:, 1])
        open_chain_motor_vel = np.column_stack(
            (
                _finite_diff(open_chain_targets[:, 0], cfg.SIM_TIMESTEP),
                _finite_diff(open_chain_targets[:, 1], cfg.SIM_TIMESTEP),
            )
        )
        left_open_chain_results = [
            open_chain_ik(
                target_pos=target_ankle,
                base_pos=left_base_pos,
                link1_length=link1_len,
                link2_length=link2_len,
                elbow_behind=arm.open_chain_elbow_behind,
            )
            for target_ankle in left_ankle_traj
        ]
        left_ik_fail_count = sum(
            result is None for result in left_open_chain_results
        )
        if left_ik_fail_count:
            print(
                f"    [FAIL] 왼쪽 open_chain IK 실패: "
                f"{left_ik_fail_count}/{N} 스텝"
            )
            sys.exit(2)
        left_open_chain_targets = np.asarray(
            left_open_chain_results,
            dtype=float,
        )
        left_open_chain_targets[:, 0] = np.unwrap(
            left_open_chain_targets[:, 0]
        )
        left_open_chain_targets[:, 1] = np.unwrap(
            left_open_chain_targets[:, 1]
        )
        left_open_chain_motor_vel = np.column_stack(
            (
                _finite_diff(
                    left_open_chain_targets[:, 0],
                    cfg.SIM_TIMESTEP,
                ),
                _finite_diff(
                    left_open_chain_targets[:, 1],
                    cfg.SIM_TIMESTEP,
                ),
            )
        )
        stand_open_chain = open_chain_ik(
            target_pos=stand_ankle,
            base_pos=base_pos_arr,
            link1_length=link1_len,
            link2_length=link2_len,
            elbow_behind=arm.open_chain_elbow_behind,
        )
        left_stand_open_chain = open_chain_ik(
            target_pos=left_stand_ankle,
            base_pos=left_base_pos,
            link1_length=link1_len,
            link2_length=link2_len,
            elbow_behind=arm.open_chain_elbow_behind,
        )
        if stand_open_chain is None or left_stand_open_chain is None:
            raise RuntimeError("중립 기립자세의 open-chain IK 계산에 실패했습니다.")
        stand_robot_targets = np.asarray(stand_open_chain, dtype=float)
        left_stand_robot_targets = np.asarray(
            left_stand_open_chain,
            dtype=float,
        )
        print(f"    양측 open-chain 발목 추종 IK 완료: {2 * N}/{2 * N} 스텝")
    elif is_remote_ee:
        pre_ik_fail_count = 0
        for target_ankle in ankle_traj:
            target_ee = compute_target_ee(target_ankle, base_pos, arm.cuff_length)
            ik_res = robot_arm_ik(
                target_pos=target_ee,
                base_pos=base_pos_arr,
                link1_length=link1_len,
                link2_length=link2_len,
                elbow_down=True,
                **arm_ik_kwargs,
            )
            if ik_res is None:
                pre_ik_fail_count += 1

        if pre_ik_fail_count > 0:
            print(
                f"    [FAIL] 사전 검증 실패: {pre_ik_fail_count}/"
                f"{len(ankle_traj)} 스텝에서 IK 해를 찾을 수 없습니다."
            )
            print("    시뮬레이션을 취소하고 조기 종료합니다.")
            sys.exit(2)
    elif is_remote_ankle:
        remote_ankle_results = [
            remote_ankle_arm_ik(
                target_pos=target_ankle,
                base_pos=base_pos_arr,
                link1_length=link1_len,
                link2_length=link2_len,
                motor_distance=arm.remote_exo_motor_dx,
                motor2_offset_z=arm.remote_exo_motor_dz,
                crank_length=arm.remote_exo_crank_length,
                upper_plate_link=arm.remote_exo_upper_link,
                joint_spacing=arm.remote_exo_joint_spacing,
                knee_behind=True,
            )
            for target_ankle in ankle_traj
        ]
        pre_ik_fail_count = sum(
            result is None for result in remote_ankle_results
        )
        if pre_ik_fail_count:
            print(
                f"    [FAIL] remote_ankle_arm IK 실패: "
                f"{pre_ik_fail_count}/{N} 스텝"
            )
            print("    link1/link2 또는 motor2 crank 도달범위를 확인하세요.")
            sys.exit(2)

        remote_ankle_targets = np.asarray(
            remote_ankle_results,
            dtype=float,
        )
        remote_ankle_targets[:, 0] = np.unwrap(
            remote_ankle_targets[:, 0]
        )
        remote_ankle_targets[:, 1] = np.unwrap(
            remote_ankle_targets[:, 1]
        )
        remote_ankle_motor_vel = np.column_stack(
            (
                _finite_diff(
                    remote_ankle_targets[:, 0],
                    cfg.SIM_TIMESTEP,
                ),
                _finite_diff(
                    remote_ankle_targets[:, 1],
                    cfg.SIM_TIMESTEP,
                ),
            )
        )
        print(f"    발목 추종 2-link IK 완료: {N}/{N} 스텝")
    elif is_remote_exo:
        remote_exo_results = [
            remote_exoskeleton_ik(
                hip_angle=hip_angle,
                knee_angle=knee_angle,
                motor_distance=arm.remote_exo_motor_dx,
                motor2_offset_z=arm.remote_exo_motor_dz,
                crank_length=arm.remote_exo_crank_length,
                upper_plate_link=arm.remote_exo_upper_link,
                joint_spacing=arm.remote_exo_joint_spacing,
            )
            for hip_angle, knee_angle in zip(
                traj.hip_angle,
                traj.knee_angle,
            )
        ]
        pre_ik_fail_count = sum(result is None for result in remote_exo_results)
        if pre_ik_fail_count:
            print(
                f"    [FAIL] remote_exoskeleton 전달기구 해석 실패: "
                f"{pre_ik_fail_count}/{N} 스텝"
            )
            print("    motor2 위치 또는 crank/rod 길이를 확인하세요.")
            sys.exit(2)

        remote_exo_targets = np.asarray(remote_exo_results, dtype=float)
        # Motor2의 ±π 표현 경계에서 목표속도가 튀지 않도록 연속화한다.
        remote_exo_targets[:, 1] = np.unwrap(remote_exo_targets[:, 1])
        remote_exo_motor_vel = np.column_stack(
            (
                _finite_diff(remote_exo_targets[:, 0], cfg.SIM_TIMESTEP),
                _finite_diff(remote_exo_targets[:, 1], cfg.SIM_TIMESTEP),
            )
        )
        print(
            "    상부 motor1/motor2 전달기구 해석 완료: "
            f"{N}/{N} 스텝"
        )
    else:
        print("    외골격은 hip/knee 목표각을 직접 사용하므로 IK 검증을 생략합니다.")

    xml_str = build_xml(
        thigh_length=seg.thigh_length,
        shank_length=seg.shank_length,
        thigh_mass=seg.thigh_mass,
        shank_mass=seg.shank_mass,
        foot_mass=seg.foot_mass,
        link1_length=link1_len,
        link2_length=link2_len,
        arm_link_mass=arm.arm_link_mass,
        base_pos=base_pos,
        cuff_length=arm.cuff_length,
        passive_damping=arm.passive_damping,
        control_mode=cfg.ROBOT_ARM.control_mode,
        sim_timestep=cfg.SIM_TIMESTEP,
        motor_offset_z=arm.motor_offset_z,
        robot_type=arm.robot_type,
        motor_distance=ik_motor_distance,
        motor2_offset_z=ik_motor2_offset_z,
        exo_lateral_offset=arm.exo_lateral_offset,
        exo_motor_mass=arm.exo_motor_mass,
        exo_link_mass=arm.exo_link_mass,
        remote_exo_motor_dx=arm.remote_exo_motor_dx,
        remote_exo_motor_dz=arm.remote_exo_motor_dz,
        remote_exo_crank_length=arm.remote_exo_crank_length,
        remote_exo_upper_link=arm.remote_exo_upper_link,
        remote_exo_joint_spacing=arm.remote_exo_joint_spacing,
        remote_exo_plate_mass=arm.remote_exo_plate_mass,
        remote_exo_coupler_mass=arm.remote_exo_coupler_mass,
        remote_exo_crank_mass=arm.remote_exo_crank_mass,
        remote_exo_rod_mass=arm.remote_exo_rod_mass,
        remote_c_top_link=ik_remote_c_top_link,
        remote_d_upper_link=ik_remote_d_upper_link,
        remote_e_motor1_joint_offset=ik_remote_e_motor1_joint_offset,
        rod_layer_offset=getattr(arm, "rod_layer_offset", 0.014),
        distal_standoff=getattr(arm, "distal_standoff", 0.030),
        hip_joint_spacing=hip_spacing,
        bilateral=True,
        walker_enabled=walker.enabled,
        walker_frame_width=walker_width,
        walker_frame_length=walker_length,
        walker_handle_height_above_hip=walker.handle_height_above_hip,
        walker_tube_radius=walker.tube_radius,
        walker_caster_radius=walker.caster_radius,
    )
    if range_limit_enabled:
        xml_str = _apply_joint_range_limits(
            xml_str,
            arm.robot_type,
            joint_ranges_deg,
        )

    # XML 저장 (디버깅용)
    xml_path = cfg.PATHS["xml_model"]
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_str, encoding="utf-8")
    print(f"    XML 저장: {xml_path}")

    model = mujoco.MjModel.from_xml_string(xml_str)
    data  = mujoco.MjData(model)

    # 관절/액추에이터 인덱스 확인
    hip_qpos_id   = _get_joint_qpos_id(model, "hip_joint")
    knee_qpos_id  = _get_joint_qpos_id(model, "knee_joint")
    m1_qpos_id    = _get_joint_qpos_id(model, "motor1_joint")
    m2_qpos_id    = _get_joint_qpos_id(model, "motor2_joint")
    hip_qvel_id   = _get_joint_qvel_id(model, "hip_joint")
    knee_qvel_id  = _get_joint_qvel_id(model, "knee_joint")
    m1_qvel_id    = _get_joint_qvel_id(model, "motor1_joint")
    m2_qvel_id    = _get_joint_qvel_id(model, "motor2_joint")
    left_hip_qpos_id = _get_joint_qpos_id(model, "hip_joint_left")
    left_knee_qpos_id = _get_joint_qpos_id(model, "knee_joint_left")
    left_m1_qpos_id = _get_joint_qpos_id(model, "motor1_joint_left")
    left_m2_qpos_id = _get_joint_qpos_id(model, "motor2_joint_left")
    left_hip_qvel_id = _get_joint_qvel_id(model, "hip_joint_left")
    left_knee_qvel_id = _get_joint_qvel_id(model, "knee_joint_left")
    left_m1_qvel_id = _get_joint_qvel_id(model, "motor1_joint_left")
    left_m2_qvel_id = _get_joint_qvel_id(model, "motor2_joint_left")
    exo_knee_qpos_id = (
        _get_joint_qpos_id(model, "exo_knee_joint")
        if is_remote_exo
        else None
    )

    hip_act_id    = _get_actuator_id(model, "hip_act")
    knee_act_id   = _get_actuator_id(model, "knee_act")
    m1_act_id     = _get_actuator_id(model, "motor1_act")
    m2_act_id     = _get_actuator_id(model, "motor2_act")
    left_hip_act_id = _get_actuator_id(model, "hip_act_left")
    left_knee_act_id = _get_actuator_id(model, "knee_act_left")
    left_m1_act_id = _get_actuator_id(model, "motor1_act_left")
    left_m2_act_id = _get_actuator_id(model, "motor2_act_left")
    passive_act_id = (
        _get_actuator_id(model, "passive_act") if is_remote_ee else None
    )

    # ── Step 5: PD 제어기 준비 ────────────────────────────────────────────
    ctrl_m1, ctrl_m2 = build_pd_controllers(cfg.PD_GAINS, arm.torque_limit)

    base_pos_arr = right_base_pos if arm.robot_type in {
        "exoskeleton",
        "open_chain",
        "remote_ankle_arm",
        "remote_exoskeleton",
    } else np.array(base_pos)
    # remote_parallelogram IK 호출에 공통으로 전달할 기하 파라미터
    arm_ik_kwargs = dict(
        d_motor=arm.motor_offset_z,
        robot_type=arm.robot_type,
        motor_distance=ik_motor_distance,
        motor2_offset_z=ik_motor2_offset_z,
        remote_c_top_link=ik_remote_c_top_link,
        remote_d_upper_link=ik_remote_d_upper_link,
        remote_e_motor1_joint_offset=ik_remote_e_motor1_joint_offset,
        distal_standoff=getattr(arm, "distal_standoff", 0.030),
    )
    if arm.robot_type == "exoskeleton":
        # 외골격 관절은 사람 hip/knee 관절과 동일한 좌표 규약을 사용한다.
        right_init = (
            stand_robot_targets
            if interactive
            else np.array([traj.hip_angle[0], traj.knee_angle[0]])
        )
        left_init = (
            left_stand_robot_targets
            if interactive
            else np.array([left_hip_angle[0], left_knee_angle[0]])
        )
        data.qpos[hip_qpos_id] = right_init[0]
        data.qpos[knee_qpos_id] = right_init[1]
        data.qpos[m1_qpos_id] = right_init[0]
        data.qpos[m2_qpos_id] = right_init[1]
        data.qpos[left_hip_qpos_id] = left_init[0]
        data.qpos[left_knee_qpos_id] = left_init[1]
        data.qpos[left_m1_qpos_id] = left_init[0]
        data.qpos[left_m2_qpos_id] = left_init[1]
        mujoco.mj_forward(model, data)
        print(
            f"    초기 외골격 각도: 오른쪽 hip/knee="
            f"{np.rad2deg(right_init[0]):.1f}°/"
            f"{np.rad2deg(right_init[1]):.1f}°, 왼쪽="
            f"{np.rad2deg(left_init[0]):.1f}°/"
            f"{np.rad2deg(left_init[1]):.1f}°"
        )
    elif is_open_chain:
        assert open_chain_targets is not None
        assert left_open_chain_targets is not None
        theta1_init, theta2_init = (
            stand_robot_targets if interactive else open_chain_targets[0]
        )
        left_theta1_init, left_theta2_init = (
            left_stand_robot_targets
            if interactive
            else left_open_chain_targets[0]
        )
        data.qpos[hip_qpos_id] = (
            stand_hip_angle if interactive else traj.hip_angle[0]
        )
        data.qpos[knee_qpos_id] = (
            stand_knee_angle if interactive else traj.knee_angle[0]
        )
        data.qpos[m1_qpos_id] = theta1_init
        data.qpos[m2_qpos_id] = theta2_init
        data.qpos[left_hip_qpos_id] = (
            stand_hip_angle if interactive else left_hip_angle[0]
        )
        data.qpos[left_knee_qpos_id] = (
            stand_knee_angle if interactive else left_knee_angle[0]
        )
        data.qpos[left_m1_qpos_id] = left_theta1_init
        data.qpos[left_m2_qpos_id] = left_theta2_init
        mujoco.mj_forward(model, data)
        print(
            "    초기 open-chain 팔: 오른쪽 motor1/motor2="
            f"{np.rad2deg(theta1_init):.1f}°/{np.rad2deg(theta2_init):.1f}°, "
            f"왼쪽={np.rad2deg(left_theta1_init):.1f}°/"
            f"{np.rad2deg(left_theta2_init):.1f}°"
        )
    elif is_remote_ankle:
        assert remote_ankle_targets is not None
        (
            theta1_init,
            theta2_init,
            plate_init,
            coupler_init,
            distal_init,
            rod_init,
        ) = remote_ankle_targets[0]

        data.qpos[hip_qpos_id] = traj.hip_angle[0]
        data.qpos[knee_qpos_id] = traj.knee_angle[0]
        data.qpos[m1_qpos_id] = theta1_init
        data.qpos[m2_qpos_id] = theta2_init
        data.qpos[_get_joint_qpos_id(model, "plate_joint")] = plate_init
        data.qpos[_get_joint_qpos_id(model, "coupler_joint")] = coupler_init
        data.qpos[_get_joint_qpos_id(model, "distal_joint")] = distal_init
        data.qpos[_get_joint_qpos_id(model, "rod_joint")] = rod_init
        mujoco.mj_forward(model, data)
        print(
            "    초기 발목 추종 팔: "
            f"motor1={np.rad2deg(theta1_init):.1f}°, "
            f"motor2={np.rad2deg(theta2_init):.1f}°, "
            f"distal={np.rad2deg(distal_init):.1f}°"
        )
    elif is_remote_exo:
        assert remote_exo_targets is not None
        (
            theta1_init,
            theta2_init,
            plate_init,
            coupler_init,
            exo_knee_init,
            rod_init,
        ) = remote_exo_targets[0]

        data.qpos[hip_qpos_id] = traj.hip_angle[0]
        data.qpos[knee_qpos_id] = traj.knee_angle[0]
        data.qpos[m1_qpos_id] = theta1_init
        data.qpos[m2_qpos_id] = theta2_init
        data.qpos[exo_knee_qpos_id] = exo_knee_init
        data.qpos[_get_joint_qpos_id(model, "plate_joint")] = plate_init
        data.qpos[_get_joint_qpos_id(model, "coupler_joint")] = coupler_init
        data.qpos[_get_joint_qpos_id(model, "rod_joint")] = rod_init
        mujoco.mj_forward(model, data)
        print(
            f"    초기 원격 외골격: hip motor="
            f"{np.rad2deg(theta1_init):.1f}°, knee motor="
            f"{np.rad2deg(theta2_init):.1f}°, knee="
            f"{np.rad2deg(exo_knee_init):.1f}°"
        )
    else:
        # EE 브라켓 끝이 발목에 닿도록 remote_parallelogram 초기 IK 계산
        target_ee_init = compute_target_ee(
            ankle_traj[0], base_pos, arm.cuff_length
        )
        ik_init = robot_arm_ik(
            target_pos=target_ee_init,
            base_pos=base_pos_arr,
            link1_length=link1_len,
            link2_length=link2_len,
            elbow_down=True,
            return_passive=True,
            **arm_ik_kwargs,
        )
        if ik_init is None:
            raise RuntimeError("remote_parallelogram 초기 IK 계산에 실패했습니다.")

        theta_passive_init = ik_init[0]
        theta1_init = ik_init[1]
        theta2_init = ik_init[2]

        passive_qpos_id = _get_joint_qpos_id(model, "base_passive_joint")
        data.qpos[passive_qpos_id] = theta_passive_init

        data.qpos[m1_qpos_id] = theta1_init
        data.qpos[m2_qpos_id] = theta2_init

        if len(ik_init) >= 8:
            # 폐루프 일관 초기화: plate / coupler / distal_D / distal_C / rod
            # D와 C 모두 passive hinge이며, C는 connect equality로 EE link의 C standoff에 결합됨
            plate_init, coupler_init, distal_d_init, distal_c_init, rod_init = ik_init[3:8]
            data.qpos[_get_joint_qpos_id(model, "plate_joint")]     = plate_init
            data.qpos[_get_joint_qpos_id(model, "coupler_joint")]   = coupler_init
            data.qpos[_get_joint_qpos_id(model, "distal_D_joint")]  = distal_d_init
            data.qpos[_get_joint_qpos_id(model, "distal_C_joint")]  = distal_c_init
            data.qpos[_get_joint_qpos_id(model, "rod_joint")]       = rod_init
        # 사람 다리 초기 관절각도 설정
        data.qpos[hip_qpos_id]  = traj.hip_angle[0]
        data.qpos[knee_qpos_id] = traj.knee_angle[0]
        mujoco.mj_forward(model, data)   # 위치 갱신
        print(
            f"    초기 IK: passive={np.rad2deg(theta_passive_init):.1f}°, "
            f"motor1={np.rad2deg(theta1_init):.1f}°, "
            f"motor2={np.rad2deg(theta2_init):.1f}°"
        )

    # ── 출력 위치 ─────────────────────────────────────────────────────────
    out_dir = cfg.PATHS["output_dir"]
    if out_dir_name:
        out_dir = out_dir / out_dir_name

    should_save_video = (render_mode == "offscreen") if save_video is None else save_video

    # ── 결과 버퍼 ─────────────────────────────────────────────────────────
    records = []

    # ── 렌더러 초기화 ─────────────────────────────────────────────────────
    viewer      = None
    renderer    = None
    frames      = []

    if render_mode == "viewer":
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            viewer.cam.azimuth   = 120
            viewer.cam.elevation = -15
            viewer.cam.distance  = 2.2
            viewer.cam.lookat[:] = [0.1, 0.0, -0.2]
        except Exception as e:
            print(f"    [WARN] viewer 실패: {e}. none 모드로 전환.")
            render_mode = "none"
    if render_mode == "offscreen" or should_save_video:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
        except Exception as e:
            print(f"    [WARN] offscreen renderer 초기화 실패: {e}")
            print("    [WARN] headless 환경이면 MUJOCO_GL=egl python src/simulate.py ... 로 실행해보세요.")
            renderer = None
            should_save_video = False

    # ── 메인 루프 ─────────────────────────────────────────────────────────
    print(f"\n[4] 시뮬레이션 시작 (render={render_mode}, cycles={n_cycles}, save_video={should_save_video}) …")
    t_start_wall = time.time()

    # IK 실패 카운터
    ik_fail_count = 0

    # UI 실행에서는 시뮬레이션 시간과 동작 진행률을 분리한다. Active는
    # swing 구간에서 환자 힘이 기준 미만일 때 현재 gait reference를 그대로
    # 유지하고, Automatic은 같은 reference를 설정 속도로 계속 진행한다.
    # cadence는 1 gait cycle = 2 steps로 환산한다.
    cycle_steps = max(2, int(round(traj.cycle_duration / cfg.SIM_TIMESTEP)))
    nominal_cadence_spm = 120.0 / traj.cycle_duration
    gait_cursor = 0.0
    gait_distance = 0.0
    runtime_state = "stand_hold"
    cadence_spm = 40.0
    patient_effort = 0.0
    initiation_threshold = 20.0
    soft_start_duration = 2.0
    sit_to_stand_duration = 3.0
    sit_to_stand_request_id = 0
    previous_sit_to_stand_request_id = 0
    sit_to_stand_elapsed = 0.0
    sit_to_stand_stage = "complete"
    sit_to_stand_progress = 100.0
    sit_to_stand_repetitions = 0
    sit_to_stand_completed = True
    sit_to_stand_hold_duration = 1.0
    active_swing_window = ""
    active_swing_authorized = False
    sit_to_stand_entry_right = np.array(
        [data.qpos[hip_qpos_id], data.qpos[knee_qpos_id]], dtype=float
    )
    sit_to_stand_entry_left = np.array(
        [data.qpos[left_hip_qpos_id], data.qpos[left_knee_qpos_id]],
        dtype=float,
    )
    previous_dynamic_right_robot_target = stand_robot_targets.copy()
    previous_dynamic_left_robot_target = left_stand_robot_targets.copy()
    right_hip_support = right_knee_support = 1.0
    left_hip_support = left_knee_support = 1.0
    tracking_alarm = False
    previous_runtime_state = runtime_state
    resume_from_safe_stop = False
    transition_elapsed = soft_start_duration
    transition_start = np.array(
        [
            data.qpos[m1_qpos_id],
            data.qpos[m2_qpos_id],
            data.qpos[left_m1_qpos_id],
            data.qpos[left_m2_qpos_id],
        ],
        dtype=float,
    )
    human_transition_start = np.array(
        [
            data.qpos[hip_qpos_id],
            data.qpos[knee_qpos_id],
            data.qpos[left_hip_qpos_id],
            data.qpos[left_knee_qpos_id],
        ],
        dtype=float,
    )

    if interactive:
        try:
            runtime_control["status"] = "running"
            runtime_control["actual_state"] = runtime_state
        except (BrokenPipeError, ConnectionError, EOFError):
            pass

    for simulation_step in range(simulation_steps):
        if interactive and simulation_step % 20 == 0:
            try:
                if bool(runtime_control.get("stop_requested", False)):
                    break
                runtime_state = str(
                    runtime_control.get("requested_state", "stand_hold")
                )
                # 이전 UI 세션에서 저장된 명칭도 새 제어 철학으로 해석한다.
                runtime_state = {
                    "walk_guided": "automatic",
                    "walk_active_assist": "active",
                }.get(runtime_state, runtime_state)
                if runtime_state not in {
                    "stand_hold",
                    "automatic",
                    "active",
                    "sit_to_stand",
                    "safe_stop",
                }:
                    runtime_state = "stand_hold"
                cadence_spm = float(runtime_control.get("cadence_spm", 40.0))
                patient_effort = float(
                    runtime_control.get("patient_effort", 0.0)
                )
                initiation_threshold = float(
                    runtime_control.get("initiation_threshold", 20.0)
                )
                sit_to_stand_duration = np.clip(
                    float(runtime_control.get("sit_to_stand_duration", 3.0)),
                    1.5,
                    8.0,
                )
                sit_to_stand_request_id = int(
                    runtime_control.get("sit_to_stand_request_id", 0)
                )
                soft_start_duration = np.clip(
                    float(runtime_control.get("soft_start_duration", 2.0)),
                    0.5,
                    5.0,
                )
                right_hip_support = np.clip(
                    float(runtime_control.get("right_hip_support", 100.0)) / 100.0,
                    0.0,
                    1.0,
                )
                right_knee_support = np.clip(
                    float(runtime_control.get("right_knee_support", 100.0)) / 100.0,
                    0.0,
                    1.0,
                )
                left_hip_support = np.clip(
                    float(runtime_control.get("left_hip_support", 100.0)) / 100.0,
                    0.0,
                    1.0,
                )
                left_knee_support = np.clip(
                    float(runtime_control.get("left_knee_support", 100.0)) / 100.0,
                    0.0,
                    1.0,
                )
            except (BrokenPipeError, ConnectionError, EOFError):
                break

        if interactive:
            step = int(gait_cursor) % cycle_steps
            gait_mode = runtime_state in {"automatic", "active"}
            reference_active = gait_mode or runtime_state == "sit_to_stand"

            # 오른쪽 보행 주기 기준 오른쪽 swing=60~100%, 50% 앞선 왼쪽
            # swing=10~50%로 둔다. Active에서는 swing 동안에만 힘 기준을
            # 적용하고, 기준 미만이면 현재 reference 위치에서 기다린다.
            gait_phase_fraction = gait_cursor / cycle_steps
            current_swing_window = (
                "left"
                if 0.10 <= gait_phase_fraction < 0.50
                else "right"
                if 0.60 <= gait_phase_fraction < 1.0
                else ""
            )
            active_swing_phase = bool(current_swing_window)
            if runtime_state != "active":
                active_swing_window = ""
                active_swing_authorized = False
            elif (
                runtime_state != previous_runtime_state
                or current_swing_window != active_swing_window
            ):
                # 매 swing 진입 때 한 번 잠그고, 힘 기준을 통과하면 해당
                # swing이 끝날 때까지 진행 허가를 유지하여 센서 chatter로
                # 궤적이 반복 정지하는 것을 막는다.
                active_swing_window = current_swing_window
                active_swing_authorized = False
            if (
                runtime_state == "active"
                and current_swing_window
                and patient_effort >= initiation_threshold
            ):
                active_swing_authorized = True
            active_gate_blocked = (
                runtime_state == "active"
                and active_swing_phase
                and not active_swing_authorized
            )
            phase_can_advance = (
                runtime_state == "automatic"
                or (runtime_state == "active" and not active_gate_blocked)
            )
            sit_to_stand_retriggered = (
                runtime_state == "sit_to_stand"
                and sit_to_stand_request_id
                != previous_sit_to_stand_request_id
            )
            if runtime_state != previous_runtime_state or sit_to_stand_retriggered:
                # 상태 전환 순간의 실제 관절각을 새 궤적의 출발점으로 사용한다.
                # Safety Stop 중 자세가 변해도 재시작 목표가 순간이동하지 않는다.
                transition_start = np.array(
                    [
                        data.qpos[m1_qpos_id],
                        data.qpos[m2_qpos_id],
                        data.qpos[left_m1_qpos_id],
                        data.qpos[left_m2_qpos_id],
                    ],
                    dtype=float,
                )
                human_transition_start = np.array(
                    [
                        data.qpos[hip_qpos_id],
                        data.qpos[knee_qpos_id],
                        data.qpos[left_hip_qpos_id],
                        data.qpos[left_knee_qpos_id],
                    ],
                    dtype=float,
                )
                transition_elapsed = 0.0
                resume_from_safe_stop = (
                    previous_runtime_state == "safe_stop"
                    and runtime_state != "safe_stop"
                )
                if runtime_state == "sit_to_stand":
                    sit_to_stand_elapsed = 0.0
                    sit_to_stand_completed = False
                    sit_to_stand_stage = "preparing"
                    sit_to_stand_progress = 0.0
                    sit_to_stand_entry_right = np.array(
                        [data.qpos[hip_qpos_id], data.qpos[knee_qpos_id]],
                        dtype=float,
                    )
                    sit_to_stand_entry_left = np.array(
                        [
                            data.qpos[left_hip_qpos_id],
                            data.qpos[left_knee_qpos_id],
                        ],
                        dtype=float,
                    )
                previous_runtime_state = runtime_state
                previous_sit_to_stand_request_id = sit_to_stand_request_id

            transition_u = float(
                np.clip(transition_elapsed / soft_start_duration, 0.0, 1.0)
            )
            # cubic smoothstep: 시작과 종료 시 목표속도가 0이므로 PD의 D항이
            # 불연속적으로 튀지 않는다.
            transition_weight = transition_u**2 * (3.0 - 2.0 * transition_u)
            transition_weight_dot = (
                6.0 * transition_u * (1.0 - transition_u) / soft_start_duration
                if transition_u < 1.0
                else 0.0
            )
            phase_increment = (
                max(0.0, cadence_spm) / nominal_cadence_spm
                * transition_weight
                if phase_can_advance
                else 0.0
            )
            velocity_scale = phase_increment
        else:
            step = simulation_step
            phase_increment = 1.0
            velocity_scale = 1.0
            gait_mode = True
            reference_active = True
            active_swing_phase = False
            active_gate_blocked = False
            transition_weight = 1.0
            transition_weight_dot = 0.0

        # ── 4-a. 사람 다리 position actuator 목표 설정 ────────────────
        if interactive and runtime_state == "sit_to_stand":
            preparation_duration = soft_start_duration
            rise_start = preparation_duration + sit_to_stand_hold_duration
            sequence_duration = rise_start + sit_to_stand_duration
            if sit_to_stand_elapsed < preparation_duration:
                sit_to_stand_stage = "preparing"
                progress, progress_dot = _quintic_profile(
                    sit_to_stand_elapsed,
                    preparation_duration,
                )
                right_start = sit_to_stand_entry_right
                left_start = sit_to_stand_entry_left
                right_end = right_sit_pose
                left_end = left_sit_pose
            elif sit_to_stand_elapsed < rise_start:
                sit_to_stand_stage = "seated_hold"
                progress = 1.0
                progress_dot = 0.0
                right_start = right_sit_pose
                left_start = left_sit_pose
                right_end = right_sit_pose
                left_end = left_sit_pose
            elif sit_to_stand_elapsed < sequence_duration:
                sit_to_stand_stage = "rising"
                progress, progress_dot = _quintic_profile(
                    sit_to_stand_elapsed - rise_start,
                    sit_to_stand_duration,
                )
                right_start = right_sit_pose
                left_start = left_sit_pose
                right_end = np.array([stand_hip_angle, stand_knee_angle])
                left_end = right_end
            else:
                sit_to_stand_stage = "complete"
                progress = 1.0
                progress_dot = 0.0
                right_start = right_end = np.array(
                    [stand_hip_angle, stand_knee_angle]
                )
                left_start = left_end = right_start
                if not sit_to_stand_completed:
                    sit_to_stand_repetitions += 1
                    sit_to_stand_completed = True

            right_sts_target = right_start + progress * (right_end - right_start)
            left_sts_target = left_start + progress * (left_end - left_start)
            right_sts_velocity = progress_dot * (right_end - right_start)
            left_sts_velocity = progress_dot * (left_end - left_start)
            human_hip_target, human_knee_target = right_sts_target
            left_human_hip_target, left_human_knee_target = left_sts_target
            human_hip_velocity, human_knee_velocity = right_sts_velocity
            left_human_hip_velocity, left_human_knee_velocity = left_sts_velocity
            sit_to_stand_progress = 100.0 * min(
                sit_to_stand_elapsed / sequence_duration,
                1.0,
            )
        elif interactive and not gait_mode:
            human_hip_target = stand_hip_angle
            human_knee_target = stand_knee_angle
            left_human_hip_target = stand_hip_angle
            left_human_knee_target = stand_knee_angle
            human_hip_velocity = human_knee_velocity = 0.0
            left_human_hip_velocity = left_human_knee_velocity = 0.0
        elif interactive:
            gait_targets = np.array(
                [
                    traj.hip_angle[step],
                    traj.knee_angle[step],
                    left_hip_angle[step],
                    left_knee_angle[step],
                ],
                dtype=float,
            )
            gait_velocities = np.array(
                [
                    hip_vel_des[step],
                    knee_vel_des[step],
                    left_hip_vel_des[step],
                    left_knee_vel_des[step],
                ],
                dtype=float,
            ) * velocity_scale
            human_targets = (
                (1.0 - transition_weight) * human_transition_start
                + transition_weight * gait_targets
            )
            human_velocities = (
                transition_weight_dot
                * (gait_targets - human_transition_start)
                + transition_weight * gait_velocities
            )
            (
                human_hip_target,
                human_knee_target,
                left_human_hip_target,
                left_human_knee_target,
            ) = human_targets
            (
                human_hip_velocity,
                human_knee_velocity,
                left_human_hip_velocity,
                left_human_knee_velocity,
            ) = human_velocities
        else:
            human_hip_target = traj.hip_angle[step]
            human_knee_target = traj.knee_angle[step]
            left_human_hip_target = left_hip_angle[step]
            left_human_knee_target = left_knee_angle[step]
            human_hip_velocity = hip_vel_des[step]
            human_knee_velocity = knee_vel_des[step]
            left_human_hip_velocity = left_hip_vel_des[step]
            left_human_knee_velocity = left_knee_vel_des[step]
        data.ctrl[hip_act_id] = human_hip_target
        data.ctrl[knee_act_id] = human_knee_target
        data.ctrl[left_hip_act_id] = left_human_hip_target
        data.ctrl[left_knee_act_id] = left_human_knee_target

        # ── 4-b. 보조 로봇 제어 및 토크 계산 ──────────────────────────
        if interactive:
            target_ankle = compute_ankle_trajectory(
                hip_angles=np.array([human_hip_target]),
                knee_angles=np.array([human_knee_target]),
                thigh_length=seg.thigh_length,
                shank_length=seg.shank_length,
                hip_pos=right_hip_global,
                scale_factor=scale,
            )[0]
            left_target_ankle = compute_ankle_trajectory(
                hip_angles=np.array([left_human_hip_target]),
                knee_angles=np.array([left_human_knee_target]),
                thigh_length=seg.thigh_length,
                shank_length=seg.shank_length,
                hip_pos=left_hip_global,
                scale_factor=scale,
            )[0]
        else:
            target_ankle = ankle_traj[step]
            left_target_ankle = left_ankle_traj[step]
        tau1_p = tau1_d = tau1_bias = np.nan
        tau2_p = tau2_d = tau2_bias = np.nan
        left_tau1_p = left_tau1_d = left_tau1_bias = np.nan
        left_tau2_p = left_tau2_d = left_tau2_bias = np.nan

        if arm.robot_type == "exoskeleton":
            # Hip/knee 동축 외골격은 사람 관절 목표각을 직접 추종한다.
            if interactive:
                theta1_raw = human_hip_target
                theta2_raw = human_knee_target
                theta1_vel_raw = human_hip_velocity
                theta2_vel_raw = human_knee_velocity
            else:
                theta1_raw = traj.hip_angle[step]
                theta2_raw = traj.knee_angle[step]
                theta1_vel_raw = hip_vel_des[step] * velocity_scale
                theta2_vel_raw = knee_vel_des[step] * velocity_scale

            if interactive:
                theta1_des = (
                    (1.0 - transition_weight) * transition_start[0]
                    + transition_weight * theta1_raw
                )
                theta2_des = (
                    (1.0 - transition_weight) * transition_start[1]
                    + transition_weight * theta2_raw
                )
                theta1_vel_des = (
                    transition_weight_dot * (theta1_raw - transition_start[0])
                    + transition_weight * theta1_vel_raw
                )
                theta2_vel_des = (
                    transition_weight_dot * (theta2_raw - transition_start[1])
                    + transition_weight * theta2_vel_raw
                )
            else:
                theta1_des = theta1_raw
                theta2_des = theta2_raw
                theta1_vel_des = theta1_vel_raw
                theta2_vel_des = theta2_vel_raw

            theta1 = data.qpos[m1_qpos_id]
            theta2 = data.qpos[m2_qpos_id]
            thetadot1 = data.qvel[m1_qvel_id]
            thetadot2 = data.qvel[m2_qvel_id]

            tau1_pd = ctrl_m1.compute(
                theta1_des, theta1, theta1_vel_des, thetadot1
            )
            tau2_pd = ctrl_m2.compute(
                theta2_des, theta2, theta2_vel_des, thetadot2
            )
            tau1 = tau1_pd + data.qfrc_bias[m1_qvel_id]
            tau2 = tau2_pd + data.qfrc_bias[m2_qvel_id]

            if interactive:
                left_theta1_raw = left_human_hip_target
                left_theta2_raw = left_human_knee_target
                left_theta1_vel_raw = left_human_hip_velocity
                left_theta2_vel_raw = left_human_knee_velocity
            else:
                left_theta1_raw = left_hip_angle[step]
                left_theta2_raw = left_knee_angle[step]
                left_theta1_vel_raw = left_hip_vel_des[step] * velocity_scale
                left_theta2_vel_raw = left_knee_vel_des[step] * velocity_scale

            if interactive:
                left_theta1_des = (
                    (1.0 - transition_weight) * transition_start[2]
                    + transition_weight * left_theta1_raw
                )
                left_theta2_des = (
                    (1.0 - transition_weight) * transition_start[3]
                    + transition_weight * left_theta2_raw
                )
                left_theta1_vel_des = (
                    transition_weight_dot
                    * (left_theta1_raw - transition_start[2])
                    + transition_weight * left_theta1_vel_raw
                )
                left_theta2_vel_des = (
                    transition_weight_dot
                    * (left_theta2_raw - transition_start[3])
                    + transition_weight * left_theta2_vel_raw
                )
            else:
                left_theta1_des = left_theta1_raw
                left_theta2_des = left_theta2_raw
                left_theta1_vel_des = left_theta1_vel_raw
                left_theta2_vel_des = left_theta2_vel_raw
            left_theta1 = data.qpos[left_m1_qpos_id]
            left_theta2 = data.qpos[left_m2_qpos_id]
            left_thetadot1 = data.qvel[left_m1_qvel_id]
            left_thetadot2 = data.qvel[left_m2_qvel_id]
            left_tau1_pd = ctrl_m1.compute(
                left_theta1_des,
                left_theta1,
                left_theta1_vel_des,
                left_thetadot1,
            )
            left_tau2_pd = ctrl_m2.compute(
                left_theta2_des,
                left_theta2,
                left_theta2_vel_des,
                left_thetadot2,
            )
            left_tau1 = left_tau1_pd + data.qfrc_bias[left_m1_qvel_id]
            left_tau2 = left_tau2_pd + data.qfrc_bias[left_m2_qvel_id]

            if interactive:
                if runtime_state == "safe_stop":
                    tau1 = tau2 = left_tau1 = left_tau2 = 0.0
                else:
                    restart_ramp = (
                        transition_weight if resume_from_safe_stop else 1.0
                    )
                    tau1 *= right_hip_support * restart_ramp
                    tau2 *= right_knee_support * restart_ramp
                    left_tau1 *= left_hip_support * restart_ramp
                    left_tau2 *= left_knee_support * restart_ramp
            data.ctrl[m1_act_id] = tau1
            data.ctrl[m2_act_id] = tau2
            data.ctrl[left_m1_act_id] = left_tau1
            data.ctrl[left_m2_act_id] = left_tau2
            mujoco.mj_step(model, data)

        elif is_open_chain:
            assert open_chain_targets is not None
            assert open_chain_motor_vel is not None
            assert left_open_chain_targets is not None
            assert left_open_chain_motor_vel is not None
            if interactive and runtime_state == "sit_to_stand":
                dynamic_target = open_chain_ik(
                    target_pos=target_ankle,
                    base_pos=base_pos_arr,
                    link1_length=link1_len,
                    link2_length=link2_len,
                    elbow_behind=arm.open_chain_elbow_behind,
                )
                if dynamic_target is None:
                    ik_fail_count += 1
                    dynamic_target = previous_dynamic_right_robot_target
                dynamic_target = np.asarray(dynamic_target, dtype=float)
                dynamic_delta = np.arctan2(
                    np.sin(dynamic_target - previous_dynamic_right_robot_target),
                    np.cos(dynamic_target - previous_dynamic_right_robot_target),
                )
                theta1_raw, theta2_raw = dynamic_target
                theta1_vel_raw, theta2_vel_raw = (
                    dynamic_delta / cfg.SIM_TIMESTEP
                )
                previous_dynamic_right_robot_target = dynamic_target.copy()
            elif interactive and not gait_mode:
                theta1_raw = stand_robot_targets[0]
                theta2_raw = stand_robot_targets[1]
                theta1_vel_raw = theta2_vel_raw = 0.0
            else:
                theta1_raw = open_chain_targets[step, 0]
                theta2_raw = open_chain_targets[step, 1]
                theta1_vel_raw = (
                    open_chain_motor_vel[step, 0] * velocity_scale
                )
                theta2_vel_raw = (
                    open_chain_motor_vel[step, 1] * velocity_scale
                )

            if interactive:
                theta1_des = (
                    (1.0 - transition_weight) * transition_start[0]
                    + transition_weight * theta1_raw
                )
                theta2_des = (
                    (1.0 - transition_weight) * transition_start[1]
                    + transition_weight * theta2_raw
                )
                theta1_vel_des = (
                    transition_weight_dot * (theta1_raw - transition_start[0])
                    + transition_weight * theta1_vel_raw
                )
                theta2_vel_des = (
                    transition_weight_dot * (theta2_raw - transition_start[1])
                    + transition_weight * theta2_vel_raw
                )
            else:
                theta1_des = theta1_raw
                theta2_des = theta2_raw
                theta1_vel_des = theta1_vel_raw
                theta2_vel_des = theta2_vel_raw

            theta1 = data.qpos[m1_qpos_id]
            theta2 = data.qpos[m2_qpos_id]
            thetadot1 = data.qvel[m1_qvel_id]
            thetadot2 = data.qvel[m2_qvel_id]

            tau1_pd = ctrl_m1.compute(
                theta1_des, theta1, theta1_vel_des, thetadot1
            )
            tau2_pd = ctrl_m2.compute(
                theta2_des, theta2, theta2_vel_des, thetadot2
            )
            tau1 = tau1_pd + data.qfrc_bias[m1_qvel_id]
            tau2 = tau2_pd + data.qfrc_bias[m2_qvel_id]

            if interactive and runtime_state == "sit_to_stand":
                left_dynamic_target = open_chain_ik(
                    target_pos=left_target_ankle,
                    base_pos=left_base_pos,
                    link1_length=link1_len,
                    link2_length=link2_len,
                    elbow_behind=arm.open_chain_elbow_behind,
                )
                if left_dynamic_target is None:
                    ik_fail_count += 1
                    left_dynamic_target = previous_dynamic_left_robot_target
                left_dynamic_target = np.asarray(
                    left_dynamic_target,
                    dtype=float,
                )
                left_dynamic_delta = np.arctan2(
                    np.sin(
                        left_dynamic_target
                        - previous_dynamic_left_robot_target
                    ),
                    np.cos(
                        left_dynamic_target
                        - previous_dynamic_left_robot_target
                    ),
                )
                left_theta1_raw, left_theta2_raw = left_dynamic_target
                left_theta1_vel_raw, left_theta2_vel_raw = (
                    left_dynamic_delta / cfg.SIM_TIMESTEP
                )
                previous_dynamic_left_robot_target = left_dynamic_target.copy()
            elif interactive and not gait_mode:
                left_theta1_raw = left_stand_robot_targets[0]
                left_theta2_raw = left_stand_robot_targets[1]
                left_theta1_vel_raw = left_theta2_vel_raw = 0.0
            else:
                left_theta1_raw = left_open_chain_targets[step, 0]
                left_theta2_raw = left_open_chain_targets[step, 1]
                left_theta1_vel_raw = (
                    left_open_chain_motor_vel[step, 0] * velocity_scale
                )
                left_theta2_vel_raw = (
                    left_open_chain_motor_vel[step, 1] * velocity_scale
                )

            if interactive:
                left_theta1_des = (
                    (1.0 - transition_weight) * transition_start[2]
                    + transition_weight * left_theta1_raw
                )
                left_theta2_des = (
                    (1.0 - transition_weight) * transition_start[3]
                    + transition_weight * left_theta2_raw
                )
                left_theta1_vel_des = (
                    transition_weight_dot
                    * (left_theta1_raw - transition_start[2])
                    + transition_weight * left_theta1_vel_raw
                )
                left_theta2_vel_des = (
                    transition_weight_dot
                    * (left_theta2_raw - transition_start[3])
                    + transition_weight * left_theta2_vel_raw
                )
            else:
                left_theta1_des = left_theta1_raw
                left_theta2_des = left_theta2_raw
                left_theta1_vel_des = left_theta1_vel_raw
                left_theta2_vel_des = left_theta2_vel_raw
            left_theta1 = data.qpos[left_m1_qpos_id]
            left_theta2 = data.qpos[left_m2_qpos_id]
            left_thetadot1 = data.qvel[left_m1_qvel_id]
            left_thetadot2 = data.qvel[left_m2_qvel_id]
            left_tau1_pd = ctrl_m1.compute(
                left_theta1_des,
                left_theta1,
                left_theta1_vel_des,
                left_thetadot1,
            )
            left_tau2_pd = ctrl_m2.compute(
                left_theta2_des,
                left_theta2,
                left_theta2_vel_des,
                left_thetadot2,
            )
            left_tau1 = left_tau1_pd + data.qfrc_bias[left_m1_qvel_id]
            left_tau2 = left_tau2_pd + data.qfrc_bias[left_m2_qvel_id]

            if interactive:
                if runtime_state == "safe_stop":
                    tau1 = tau2 = left_tau1 = left_tau2 = 0.0
                else:
                    restart_ramp = (
                        transition_weight if resume_from_safe_stop else 1.0
                    )
                    tau1 *= right_hip_support * restart_ramp
                    tau2 *= right_knee_support * restart_ramp
                    left_tau1 *= left_hip_support * restart_ramp
                    left_tau2 *= left_knee_support * restart_ramp
            data.ctrl[m1_act_id] = tau1
            data.ctrl[m2_act_id] = tau2
            data.ctrl[left_m1_act_id] = left_tau1
            data.ctrl[left_m2_act_id] = left_tau2
            mujoco.mj_step(model, data)

        elif is_remote_ankle:
            assert remote_ankle_targets is not None
            assert remote_ankle_motor_vel is not None
            theta1_des = remote_ankle_targets[step, 0]
            theta2_des = remote_ankle_targets[step, 1]
            theta1_vel_des = remote_ankle_motor_vel[step, 0]
            theta2_vel_des = remote_ankle_motor_vel[step, 1]

            theta1 = data.qpos[m1_qpos_id]
            theta2 = data.qpos[m2_qpos_id]
            thetadot1 = data.qvel[m1_qvel_id]
            thetadot2 = data.qvel[m2_qvel_id]

            tau1_pd = ctrl_m1.compute(
                theta1_des,
                theta1,
                theta1_vel_des,
                thetadot1,
            )
            tau2_pd = ctrl_m2.compute(
                theta2_des,
                theta2,
                theta2_vel_des,
                thetadot2,
            )
            tau1 = tau1_pd + data.qfrc_bias[m1_qvel_id]
            tau2 = tau2_pd + data.qfrc_bias[m2_qvel_id]

            data.ctrl[m1_act_id] = tau1
            data.ctrl[m2_act_id] = tau2
            mujoco.mj_step(model, data)

        elif is_remote_exo:
            assert remote_exo_targets is not None
            assert remote_exo_motor_vel is not None
            theta1_des = remote_exo_targets[step, 0]
            theta2_des = remote_exo_targets[step, 1]
            theta1_vel_des = remote_exo_motor_vel[step, 0]
            theta2_vel_des = remote_exo_motor_vel[step, 1]

            theta1 = data.qpos[m1_qpos_id]
            theta2 = data.qpos[m2_qpos_id]
            thetadot1 = data.qvel[m1_qvel_id]
            thetadot2 = data.qvel[m2_qvel_id]

            tau1_pd = ctrl_m1.compute(
                theta1_des, theta1, theta1_vel_des, thetadot1
            )
            tau2_pd = ctrl_m2.compute(
                theta2_des, theta2, theta2_vel_des, thetadot2
            )
            tau1 = tau1_pd + data.qfrc_bias[m1_qvel_id]
            tau2 = tau2_pd + data.qfrc_bias[m2_qvel_id]

            data.ctrl[m1_act_id] = tau1
            data.ctrl[m2_act_id] = tau2
            mujoco.mj_step(model, data)

        elif cfg.ROBOT_ARM.control_mode in ["ik_pd", "robot_drives_human"]:
            # [IK + PD 제어 모드 또는 로봇 주도 수동 보행 모드]
            # 센서 피드백에 의한 고주파 노이즈 유입을 막기 위해 
            # 수학적으로 부드러운 원본 해석 궤적(ankle_traj)을 목표 위치로 직접 사용
            ik_ankle_target = ankle_traj[step].copy()

            # 로봇팔 EE 목표: 브라켓 끝이 발목에 정확히 닿도록 기하학적 보정
            target_ee = compute_target_ee(ik_ankle_target, base_pos, cfg.ROBOT_ARM.cuff_length)

            ik_result = robot_arm_ik(
                target_pos=target_ee,
                base_pos=base_pos_arr,
                link1_length=link1_len,
                link2_length=link2_len,
                elbow_down=True,
                **arm_ik_kwargs,
            )
            if ik_result is None:
                ik_fail_count += 1
                theta1_des = data.qpos[m1_qpos_id]
                theta2_des = data.qpos[m2_qpos_id]
                theta_passive_des = float(_get_sensor(model, data, "s_passive")[0])
                theta1_vel_des = 0.0
                theta2_vel_des = 0.0
            else:
                theta_passive_des, theta1_des, theta2_des = ik_result[:3]
                if step == 0 or step == N - 1:
                    theta1_vel_des = theta2_vel_des = 0.0
                else:
                    prev_ee = compute_target_ee(ankle_traj[step-1], base_pos, cfg.ROBOT_ARM.cuff_length)
                    next_ee = compute_target_ee(ankle_traj[min(step+1, N-1)], base_pos, cfg.ROBOT_ARM.cuff_length)
                    prev = robot_arm_ik(
                        prev_ee, base_pos_arr, link1_len, link2_len,
                        elbow_down=True, **arm_ik_kwargs,
                    )
                    next_ = robot_arm_ik(
                        next_ee, base_pos_arr, link1_len, link2_len,
                        elbow_down=True, **arm_ik_kwargs,
                    )
                    if prev and next_:
                        theta1_vel_des = (next_[1] - prev[1]) / (2 * cfg.SIM_TIMESTEP)
                        theta2_vel_des = (next_[2] - prev[2]) / (2 * cfg.SIM_TIMESTEP)
                    else:
                        theta1_vel_des = theta2_vel_des = 0.0

            theta1  = data.qpos[m1_qpos_id]
            theta2  = data.qpos[m2_qpos_id]
            thetadot1 = data.qvel[m1_qvel_id]
            thetadot2 = data.qvel[m2_qvel_id]

            # PD 제어 토크 + 중력 및 코리올리 보상(Feedforward)
            tau1_pd = ctrl_m1.compute(theta1_des, theta1, theta1_vel_des, thetadot1)
            tau2_pd = ctrl_m2.compute(theta2_des, theta2, theta2_vel_des, thetadot2)
            
            tau1 = tau1_pd + data.qfrc_bias[m1_qvel_id]
            tau2 = tau2_pd + data.qfrc_bias[m2_qvel_id]

            data.ctrl[m1_act_id] = tau1
            data.ctrl[m2_act_id] = tau2
            data.ctrl[passive_act_id] = theta_passive_des

            # 시뮬레이션 스텝 진행
            mujoco.mj_step(model, data)
            
        elif cfg.ROBOT_ARM.control_mode == "constraint_id":
            # [구속 + 역동역학(Inverse Dynamics) 모드]
            # 구속(weld)으로 인해 사람 다리가 움직일 때 로봇팔은 자동으로 끌려감.
            data.ctrl[m1_act_id] = 0.0
            data.ctrl[m2_act_id] = 0.0
            data.ctrl[passive_act_id] = 0.0
            
            # 1) 시뮬레이션 스텝을 진행하여 현재 step의 속도, 가속도가 발생하도록 함
            mujoco.mj_step(model, data)
            
            # 2) Inverse Dynamics 계산: 위에서 발생한 움직임을 만들기 위해 각 관절이 내야하는 힘 산출
            mujoco.mj_inverse(model, data)
            
            # 3) 산출된 필요 토크 읽기
            tau1 = data.qfrc_inverse[m1_qvel_id]
            tau2 = data.qfrc_inverse[m2_qvel_id]
            
            # 기록용 변수 채우기 (목표와 실제가 구속으로 인해 동일함)
            theta1 = data.qpos[m1_qpos_id]
            theta2 = data.qpos[m2_qpos_id]
            theta1_des = theta1
            theta2_des = theta2
            thetadot1 = data.qvel[m1_qvel_id]
            thetadot2 = data.qvel[m2_qvel_id]
            theta1_vel_des = thetadot1
            theta2_vel_des = thetadot2

        # PD 모드의 명령토크 성분을 제어 계산 시점의 상태로 분해한다.
        # 기록 후 상태로 역산하면 1 timestep 위상차가 섞이므로 여기서 계산한다.
        if cfg.ROBOT_ARM.control_mode != "constraint_id" or arm.robot_type in {
            "exoskeleton",
            "open_chain",
            "remote_ankle_arm",
            "remote_exoskeleton",
        }:
            tau1_p = cfg.PD_GAINS.Kp_motor1 * (theta1_des - theta1)
            tau1_d = cfg.PD_GAINS.Kd_motor1 * (theta1_vel_des - thetadot1)
            tau1_bias = tau1 - tau1_pd
            tau2_p = cfg.PD_GAINS.Kp_motor2 * (theta2_des - theta2)
            tau2_d = cfg.PD_GAINS.Kd_motor2 * (theta2_vel_des - thetadot2)
            tau2_bias = tau2 - tau2_pd
            if arm.robot_type in {"exoskeleton", "open_chain"}:
                left_tau1_p = cfg.PD_GAINS.Kp_motor1 * (
                    left_theta1_des - left_theta1
                )
                left_tau1_d = cfg.PD_GAINS.Kd_motor1 * (
                    left_theta1_vel_des - left_thetadot1
                )
                left_tau1_bias = left_tau1 - left_tau1_pd
                left_tau2_p = cfg.PD_GAINS.Kp_motor2 * (
                    left_theta2_des - left_theta2
                )
                left_tau2_d = cfg.PD_GAINS.Kd_motor2 * (
                    left_theta2_vel_des - left_thetadot2
                )
                left_tau2_bias = left_tau2 - left_tau2_pd

        # ── 4-e. 센서 읽기 ─────────────────────────────────────────────
        # 제어 계산에 사용한 theta/thetadot은 mj_step 이전 상태이다. 위치
        # 추종값과 같은 시점으로 비교할 수 있도록 실제 상태는 step 이후에
        # 다시 읽어서 기록한다.
        theta1_meas = float(data.qpos[m1_qpos_id])
        theta2_meas = float(data.qpos[m2_qpos_id])
        thetadot1_meas = float(data.qvel[m1_qvel_id])
        thetadot2_meas = float(data.qvel[m2_qvel_id])
        hip_angle_meas = float(data.qpos[hip_qpos_id])
        knee_angle_meas = float(data.qpos[knee_qpos_id])
        ankle_pos_meas = _get_sensor(model, data, "s_ankle_pos").copy()
        ee_pos_meas    = _get_sensor(model, data, "s_ee_pos").copy()
        left_theta1_meas = float(data.qpos[left_m1_qpos_id])
        left_theta2_meas = float(data.qpos[left_m2_qpos_id])
        left_thetadot1_meas = float(data.qvel[left_m1_qvel_id])
        left_thetadot2_meas = float(data.qvel[left_m2_qvel_id])
        left_hip_angle_meas = float(data.qpos[left_hip_qpos_id])
        left_knee_angle_meas = float(data.qpos[left_knee_qpos_id])
        left_ankle_pos_meas = _get_sensor(
            model,
            data,
            "s_ankle_pos_left",
        ).copy()
        left_ee_pos_meas = _get_sensor(model, data, "s_ee_pos_left").copy()
        passive_angle = (
            float(_get_sensor(model, data, "s_passive")[0])
            if arm.robot_type in {
                "remote_ankle_arm",
                "remote_exoskeleton",
                "remote_parallelogram",
            }
            else 0.0
        )
        exo_knee_angle = (
            float(data.qpos[exo_knee_qpos_id])
            if is_remote_exo
            else float(data.qpos[knee_qpos_id])
        )

        if interactive:
            cycle_idx = int(gait_distance / cycle_steps)
            norm_time = 100.0 * gait_cursor / cycle_steps
        else:
            cycle_idx = int(data.time / traj.cycle_duration) if traj.cycle_duration > 0 else 0
            norm_time = (data.time % traj.cycle_duration) / traj.cycle_duration * 100 if traj.cycle_duration > 0 else 0

        record = {
            "time":               data.time,
            "cycle_idx":          cycle_idx,
            "norm_time":          norm_time,
            "ankle_target_x":     target_ankle[0],
            "ankle_target_y":     target_ankle[1],
            "ankle_target_z":     target_ankle[2],
            "ankle_actual_x":     ankle_pos_meas[0],
            "ankle_actual_y":     ankle_pos_meas[1],
            "ankle_actual_z":     ankle_pos_meas[2],
            "ee_x":               ee_pos_meas[0],
            "ee_y":               ee_pos_meas[1],
            "ee_z":               ee_pos_meas[2],
            "hip_theta_des":      human_hip_target,
            "hip_theta":          hip_angle_meas,
            "knee_theta_des":     human_knee_target,
            "knee_theta":         knee_angle_meas,
            "motor1_theta_des":   theta1_des,
            "motor1_theta":       theta1_meas,
            "motor1_thetadot":    thetadot1_meas,
            "motor1_torque":      tau1,
            "motor1_torque_p":    tau1_p,
            "motor1_torque_d":    tau1_d,
            "motor1_torque_bias": tau1_bias,
            "motor2_theta_des":   theta2_des,
            "motor2_theta":       theta2_meas,
            "motor2_thetadot":    thetadot2_meas,
            "motor2_torque":      tau2,
            "motor2_torque_p":    tau2_p,
            "motor2_torque_d":    tau2_d,
            "motor2_torque_bias": tau2_bias,
            "passive_angle":      passive_angle,
            "exo_knee_theta":     exo_knee_angle,
            # 명시적인 side prefix. 위 무접미사 열은 기존 분석 코드와의
            # 호환성을 위해 동일한 오른쪽 값을 계속 제공한다.
            "right_ankle_target_x": target_ankle[0],
            "right_ankle_target_y": target_ankle[1],
            "right_ankle_target_z": target_ankle[2],
            "right_ankle_actual_x": ankle_pos_meas[0],
            "right_ankle_actual_y": ankle_pos_meas[1],
            "right_ankle_actual_z": ankle_pos_meas[2],
            "right_ee_x": ee_pos_meas[0],
            "right_ee_y": ee_pos_meas[1],
            "right_ee_z": ee_pos_meas[2],
            "right_hip_theta_des": human_hip_target,
            "right_hip_theta": hip_angle_meas,
            "right_knee_theta_des": human_knee_target,
            "right_knee_theta": knee_angle_meas,
            "right_motor1_theta_des": theta1_des,
            "right_motor1_theta": theta1_meas,
            "right_motor1_thetadot": thetadot1_meas,
            "right_motor1_torque": tau1,
            "right_motor2_theta_des": theta2_des,
            "right_motor2_theta": theta2_meas,
            "right_motor2_thetadot": thetadot2_meas,
            "right_motor2_torque": tau2,
            "left_ankle_target_x": left_target_ankle[0],
            "left_ankle_target_y": left_target_ankle[1],
            "left_ankle_target_z": left_target_ankle[2],
            "left_ankle_actual_x": left_ankle_pos_meas[0],
            "left_ankle_actual_y": left_ankle_pos_meas[1],
            "left_ankle_actual_z": left_ankle_pos_meas[2],
            "left_ee_x": left_ee_pos_meas[0],
            "left_ee_y": left_ee_pos_meas[1],
            "left_ee_z": left_ee_pos_meas[2],
            "left_hip_theta_des": left_human_hip_target,
            "left_hip_theta": left_hip_angle_meas,
            "left_knee_theta_des": left_human_knee_target,
            "left_knee_theta": left_knee_angle_meas,
            "left_motor1_theta_des": left_theta1_des,
            "left_motor1_theta": left_theta1_meas,
            "left_motor1_thetadot": left_thetadot1_meas,
            "left_motor1_torque": left_tau1,
            "left_motor1_torque_p": left_tau1_p,
            "left_motor1_torque_d": left_tau1_d,
            "left_motor1_torque_bias": left_tau1_bias,
            "left_motor2_theta_des": left_theta2_des,
            "left_motor2_theta": left_theta2_meas,
            "left_motor2_thetadot": left_thetadot2_meas,
            "left_motor2_torque": left_tau2,
            "left_motor2_torque_p": left_tau2_p,
            "left_motor2_torque_d": left_tau2_d,
            "left_motor2_torque_bias": left_tau2_bias,
            "therapy_mode": runtime_state if interactive else "batch_tracking",
            "cadence_spm": cadence_spm if interactive else nominal_cadence_spm,
            "patient_effort_percent": patient_effort if interactive else np.nan,
            "active_effort_threshold_percent": (
                initiation_threshold if interactive else np.nan
            ),
            "active_swing_phase": (
                active_swing_phase if interactive else False
            ),
            "active_gate_blocked": (
                active_gate_blocked if interactive else False
            ),
            "sit_to_stand_stage": (
                sit_to_stand_stage if interactive else ""
            ),
            "sit_to_stand_progress_percent": (
                sit_to_stand_progress if interactive else np.nan
            ),
            "sit_to_stand_rise_duration_s": (
                sit_to_stand_duration if interactive else np.nan
            ),
            "right_hip_support_percent": 100.0 * right_hip_support,
            "right_knee_support_percent": 100.0 * right_knee_support,
            "left_hip_support_percent": 100.0 * left_hip_support,
            "left_knee_support_percent": 100.0 * left_knee_support,
            "soft_start_percent": (
                100.0 * transition_weight if interactive else 100.0
            ),
            "right_hip_min_deg": (
                joint_ranges_deg["right_hip"][0]
                if range_limit_enabled
                else np.nan
            ),
            "right_hip_max_deg": (
                joint_ranges_deg["right_hip"][1]
                if range_limit_enabled
                else np.nan
            ),
            "right_knee_min_deg": (
                joint_ranges_deg["right_knee"][0]
                if range_limit_enabled
                else np.nan
            ),
            "right_knee_max_deg": (
                joint_ranges_deg["right_knee"][1]
                if range_limit_enabled
                else np.nan
            ),
            "left_hip_min_deg": (
                joint_ranges_deg["left_hip"][0]
                if range_limit_enabled
                else np.nan
            ),
            "left_hip_max_deg": (
                joint_ranges_deg["left_hip"][1]
                if range_limit_enabled
                else np.nan
            ),
            "left_knee_min_deg": (
                joint_ranges_deg["left_knee"][0]
                if range_limit_enabled
                else np.nan
            ),
            "left_knee_max_deg": (
                joint_ranges_deg["left_knee"][1]
                if range_limit_enabled
                else np.nan
            ),
        }
        # UI 세션은 1 kHz로 제어하되 로그는 100 Hz로 저장하여 장시간
        # 세션의 메모리와 CSV 크기를 제한한다.
        if not interactive or simulation_step % 10 == 0:
            records.append(record)

        # 조기 종료 (Dynamic Tracking Failure): 시뮬레이션 초반 0.5초가 지난 후, 순간 오차가 60mm 이상이면 실패 처리
        tracking_check_enabled = (
            not interactive
            or (
                reference_active
                and runtime_state != "safe_stop"
                and transition_weight >= 0.999
            )
        )
        if data.time > 0.5 and tracking_check_enabled:
            right_err_dist = np.hypot(
                ankle_pos_meas[0] - target_ankle[0],
                ankle_pos_meas[2] - target_ankle[2],
            )
            left_err_dist = np.hypot(
                left_ankle_pos_meas[0] - left_target_ankle[0],
                left_ankle_pos_meas[2] - left_target_ankle[2],
            )
            err_dist = max(right_err_dist, left_err_dist)
            if err_dist > 0.06 and not tracking_alarm:
                failed_side = "오른쪽" if right_err_dist >= left_err_dist else "왼쪽"
                print(
                    f"    [FAIL] 동역학적 추종 실패: t={data.time:.2f}s "
                    f"{failed_side} 발목 오차가 {err_dist*1000:.1f}mm "
                    "(> 60mm) 입니다."
                )
                if is_remote_exo:
                    print(
                        "    remote_exoskeleton 상태: "
                        f"hip={np.rad2deg(data.qpos[hip_qpos_id]):.1f}°/"
                        f"{np.rad2deg(traj.hip_angle[step]):.1f}°, "
                        f"knee={np.rad2deg(data.qpos[knee_qpos_id]):.1f}°/"
                        f"{np.rad2deg(traj.knee_angle[step]):.1f}°, "
                        f"M1={np.rad2deg(theta1):.1f}°/"
                        f"{np.rad2deg(theta1_des):.1f}°, "
                        f"M2={np.rad2deg(theta2):.1f}°/"
                        f"{np.rad2deg(theta2_des):.1f}°, "
                        f"plate={np.rad2deg(passive_angle):.1f}°/"
                        f"{np.rad2deg(remote_exo_targets[step, 2]):.1f}°, "
                        f"exo knee={np.rad2deg(exo_knee_angle):.1f}°"
                    )
                if interactive:
                    tracking_alarm = True
                    runtime_state = "safe_stop"
                    try:
                        runtime_control["requested_state"] = "safe_stop"
                        runtime_control["actual_state"] = "safe_stop"
                        runtime_control["alarm"] = (
                            f"{failed_side} 발목 추종 오차 {err_dist * 1000:.0f} mm"
                        )
                    except (BrokenPipeError, ConnectionError, EOFError):
                        pass
                    print("    UI 안전정지로 전환합니다.")
                else:
                    print("    시뮬레이션을 취소하고 조기 종료합니다.")
                    sys.exit(3)

        if interactive:
            if gait_mode:
                gait_distance += phase_increment
                gait_cursor = gait_distance % cycle_steps
            if runtime_state == "sit_to_stand":
                sit_to_stand_elapsed += cfg.SIM_TIMESTEP
            transition_elapsed += cfg.SIM_TIMESTEP
            if simulation_step % 50 == 0:
                try:
                    runtime_control["status"] = "running"
                    runtime_control["actual_state"] = runtime_state
                    runtime_control["sim_time"] = float(data.time)
                    runtime_control["gait_phase_pct"] = float(
                        sit_to_stand_progress
                        if runtime_state == "sit_to_stand"
                        else 100.0 * gait_cursor / cycle_steps
                    )
                    runtime_control["step_count"] = int(
                        gait_distance / (0.5 * cycle_steps)
                    )
                    runtime_control["soft_start_pct"] = float(
                        100.0 * transition_weight
                    )
                    runtime_control["active_gate_blocked"] = bool(
                        active_gate_blocked
                    )
                    runtime_control["active_swing_phase"] = bool(
                        active_swing_phase
                    )
                    runtime_control["sit_to_stand_stage"] = (
                        sit_to_stand_stage
                    )
                    runtime_control["sit_to_stand_repetitions"] = int(
                        sit_to_stand_repetitions
                    )
                    runtime_control["right_hip_deg"] = float(
                        np.rad2deg(hip_angle_meas)
                    )
                    runtime_control["right_knee_deg"] = float(
                        np.rad2deg(knee_angle_meas)
                    )
                    runtime_control["left_hip_deg"] = float(
                        np.rad2deg(left_hip_angle_meas)
                    )
                    runtime_control["left_knee_deg"] = float(
                        np.rad2deg(left_knee_angle_meas)
                    )
                    runtime_control["right_hip_torque"] = float(tau1)
                    runtime_control["right_knee_torque"] = float(tau2)
                    runtime_control["left_hip_torque"] = float(left_tau1)
                    runtime_control["left_knee_torque"] = float(left_tau2)
                except (BrokenPipeError, ConnectionError, EOFError):
                    break

        # ── 4-f. 렌더 ─────────────────────────────────────────────────
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            # 실시간 속도 맞추기
            elapsed = time.time() - t_start_wall
            desired  = data.time
            if desired - elapsed > 0:
                time.sleep(desired - elapsed)

        if renderer is not None:
            if simulation_step % max(1, int(1.0 / (cfg.VIDEO_FPS * cfg.SIM_TIMESTEP))) == 0:
                renderer.update_scene(data, camera=camera_name)
                frames.append(renderer.render().copy())

    # ── IK 실패 통계 ──────────────────────────────────────────────────────
    if ik_fail_count > 0:
        print(f"\n    [WARN] IK 실패 스텝 수: {ik_fail_count}/{N} "
              f"({100*ik_fail_count/N:.1f}%) — 로봇팔 링크 길이 확인 권장.")

    # ── 뷰어 유지 ─────────────────────────────────────────────────────────
    if viewer is not None:
        if interactive:
            viewer.close()
        else:
            print("\n    [Info] 시뮬레이션 완료. 뷰어 창을 닫으면 프로그램이 종료됩니다.")
            while viewer.is_running():
                time.sleep(0.1)
            viewer.close()

    mode_name = arm.control_mode
    if should_save_video and frames:
        video_out = Path(video_path) if video_path is not None else out_dir / "simulation.mp4"
        _save_video(frames, video_out, cfg.VIDEO_FPS)
    elif should_save_video:
        print("    [WARN] 저장할 영상 프레임이 없습니다.")

    # ── Step 6: 결과 저장 ─────────────────────────────────────────────────
    print("\n[5] 결과 저장 …")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(records)
    csv_out = out_dir / f"torque_profile.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"    CSV 저장: {csv_out}")

    # ── Step 7: 선택 후처리 ───────────────────────────────────────────────
    if plot_all is not None:
        print("\n[6] 후처리 …")
        plot_dir = out_dir / f"plots_{arm.robot_type}_{mode_name}"
        plot_all(csv_out, plot_dir)
    else:
        print("\n[6] 후처리 건너뜀: plot_results.py 없음")

    t_wall = time.time() - t_start_wall
    print(f"\n시뮬레이션 완료. 총 경과 시간: {t_wall:.1f}s")
    print("=" * 60)
    if interactive:
        try:
            runtime_control["status"] = "finished"
            runtime_control["actual_state"] = "finished"
        except (BrokenPipeError, ConnectionError, EOFError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 영상 저장 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _save_video(frames: list, path: Path, fps: int) -> None:
    """numpy 프레임 리스트를 mp4로 저장한다 (imageio 필요)."""
    try:
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(path), fps=fps) as writer:
            for f in frames:
                writer.append_data(f)
        print(f"    영상 저장: {path}")
    except ImportError:
        print("    [WARN] imageio 미설치. pip install imageio[ffmpeg] 로 설치하세요.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="소아 보행 보조 로봇 MuJoCo 시뮬레이션")
    parser.add_argument(
        "--render",
        choices=["viewer", "offscreen", "none"],
        default=cfg.RENDER_MODE,
        help="렌더 모드 (default: %(default)s)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=cfg.SIM_N_CYCLES,
        help="보행 사이클 반복 횟수 (default: %(default)s)",
    )
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument(
        "--hip-spacing",
        type=float,
        default=None,
        help="좌우 hip joint center 간격[m]. 생략하면 키의 16%%.",
    )
    parser.add_argument(
        "--robot_type",
        choices=[
            "exoskeleton",
            "open_chain",
        ],
        default=None,
        help="보조 로봇 구조",
    )
    parser.add_argument("--kp", type=float, default=None, help="두 모터에 같은 Kp를 적용")
    parser.add_argument("--kd", type=float, default=None, help="두 모터에 같은 Kd를 적용")
    parser.add_argument("--kp_motor1", type=float, default=None, help="Motor 1 Kp override")
    parser.add_argument("--kd_motor1", type=float, default=None, help="Motor 1 Kd override")
    parser.add_argument("--kp_motor2", type=float, default=None, help="Motor 2 Kp override")
    parser.add_argument("--kd_motor2", type=float, default=None, help="Motor 2 Kd override")
    parser.add_argument("--link1", type=float, default=None)
    parser.add_argument("--link2", type=float, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="viewer/none 모드에서도 mp4 영상을 저장합니다. offscreen 모드는 기본 저장.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="저장할 mp4 경로. 생략하면 output 디렉터리의 simulation.mp4.",
    )
    parser.add_argument(
        "--camera",
        choices=["fixed", "presentation"],
        default="fixed",
        help="offscreen 영상 카메라: fixed=시상면, presentation=모터 확인용 사선",
    )
    parser.add_argument(
        "--gait-data",
        type=Path,
        default=None,
        help="보행 CSV 경로. 생략하면 3–4세 Comfortable 오른쪽 평균 궤적.",
    )
    parser.add_argument(
        "--cycle-duration",
        type=float,
        default=None,
        help=(
            "시뮬레이션 1 gait cycle 시간[s]. 기본 보행 입력에서는 "
            f"{cfg.DEFAULT_GAIT_CYCLE_DURATION:.1f}s, 별도 CSV에서는 원본 시간 사용."
        ),
    )
    
    args = parser.parse_args()
    run_simulation(
        render_mode=args.render, 
        n_cycles=args.cycles,
        subject_id=args.subject,
        speed=args.speed,
        height=args.height,
        weight=args.weight,
        hip_joint_spacing=args.hip_spacing,
        robot_type=args.robot_type,
        kp_val=args.kp,
        kd_val=args.kd,
        kp_motor1=args.kp_motor1,
        kd_motor1=args.kd_motor1,
        kp_motor2=args.kp_motor2,
        kd_motor2=args.kd_motor2,
        link1=args.link1,
        link2=args.link2,
        out_dir_name=args.outdir,
        save_video=True if args.save_video else None,
        video_path=args.video_path,
        camera_name=args.camera,
        gait_data_path=args.gait_data,
        cycle_duration=args.cycle_duration,
    )


if __name__ == "__main__":
    main()
