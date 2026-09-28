"""
gait_loader.py
==============
gait_data.csv 파싱 및 보간 모듈.

지원 모드:
  - "angles" : hip_angle, knee_angle (deg) 직접 로드 → rad 변환
  - "positions": hip/knee/ankle 의 3D 위치 시계열 로드 → FK로 관절각 추출

CSV 컬럼 매핑은 config.CSV_COLUMN_MAP 에서 설정한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from dataclasses import dataclass


@dataclass
class GaitTrajectory:
    """파싱·보간 후 출력되는 보행 궤적 데이터."""
    time:        np.ndarray   # [s] (N,)
    hip_angle:   np.ndarray   # [rad] (N,)  양수 = flexion
    knee_angle:  np.ndarray   # [rad] (N,)  양수 = flexion
    cycle_duration: float = 0.0 # [s] 1 사이클 길이
    source_cycle_duration: float = 0.0  # [s] CSV에 기록된 원본 1사이클 길이
    temporal_scale: float = 1.0  # simulation/source 시간비


def _close_cycle(values: np.ndarray) -> np.ndarray:
    """
    반복 보행 사이클에서 마지막 샘플이 첫 샘플로 자연스럽게 이어지도록
    end-start 차이를 전체 phase에 선형으로 분산한다.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.copy()
    alpha = np.linspace(0.0, 1.0, len(values))
    return values - alpha * (values[-1] - values[0])


def load_gait_data(
    csv_path: str,
    col_map: dict,
    dt_csv: float,
    target_dt: float,
    n_cycles: int = 1,
    subject_id: str | None = None,
    speed: float | None = None,
    cycle_duration_override: float | None = None,
) -> GaitTrajectory:
    """
    CSV에서 보행 데이터를 로드하고, 시뮬레이션 timestep에 맞게 보간한다.

    Parameters
    ----------
    csv_path   : CSV 파일 경로.
    col_map    : config.CSV_COLUMN_MAP.
    dt_csv     : CSV 의 원본 샘플링 주기 [s].
    target_dt  : 시뮬레이션 timestep [s].
    n_cycles   : 반복할 보행 사이클 수.
    cycle_duration_override : 지정하면 관절각 파형은 유지하고 1사이클 시간만
        해당 값[s]으로 늘이거나 줄인다.

    Returns
    -------
    GaitTrajectory (보간 완료, n_cycles 반복 포함)

    Notes
    -----
    - 각도 단위: CSV = degrees → 출력 = radians.
    - 보간: cubic spline (scipy.interpolate.interp1d kind='cubic').
    - 반복: 마지막 샘플 ≈ 첫 샘플(주기성)이라 가정하고 단순 타일링 후 재보간.
    """
    df = pd.read_csv(csv_path)
    
    # Subject나 Speed가 주어지지 않았을 때 전체 데이터를 불러와서 생기는 데이터 튐 현상 방지
    if subject_id is None and "Subject" in df.columns:
        subject_id = df["Subject"].iloc[0]
        print(f"    [Info] subject_id가 지정되지 않아 기본값({subject_id})을 사용합니다.")
    if speed is None and "Speed" in df.columns:
        speed = df[df["Subject"] == subject_id]["Speed"].iloc[0]
        print(f"    [Info] speed가 지정되지 않아 기본값({speed})을 사용합니다.")

    if subject_id is not None and "Subject" in df.columns:
        df = df[df["Subject"] == subject_id]
    if speed is not None and "Speed" in df.columns:
        df = df[np.isclose(df["Speed"], speed)]
    if len(df) == 0:
        raise ValueError(f"No data found for Subject={subject_id}, Speed={speed}")
    if "Phase" in df.columns:
        df = df.sort_values("Phase")

    mode = col_map["mode"]

    if mode == "angles":
        hip_deg  = _close_cycle(df[col_map["hip_angle"]].to_numpy(dtype=float))
        knee_deg = _close_cycle(df[col_map["knee_angle"]].to_numpy(dtype=float))

        n_samples = len(hip_deg)

        # 시간축 생성
        if col_map["time"] and col_map["time"] in df.columns:
            t_csv = df[col_map["time"]].to_numpy(dtype=float)
        elif "Phase" in df.columns:
            phase = df["Phase"].to_numpy(dtype=float)
            t_csv = (phase - phase[0]) / (phase[-1] - phase[0]) * ((n_samples - 1) * dt_csv)
        else:
            t_csv = np.arange(n_samples) * dt_csv

        t_csv = t_csv - t_csv[0]
        source_cycle_duration = float(t_csv[-1])
        if source_cycle_duration <= 0:
            raise ValueError("Gait time/phase axis must have a positive duration.")

        if cycle_duration_override is not None:
            if cycle_duration_override <= 0:
                raise ValueError("cycle_duration_override must be positive.")
            cycle_duration = float(cycle_duration_override)
        else:
            cycle_duration = source_cycle_duration
        temporal_scale = cycle_duration / source_cycle_duration
        t_csv = t_csv * temporal_scale

        # 1 사이클 periodic cubic 보간. t_fine은 endpoint를 제외해 타일링 시 중복을 피한다.
        f_hip  = CubicSpline(t_csv, hip_deg,  bc_type="periodic")
        f_knee = CubicSpline(t_csv, knee_deg, bc_type="periodic")

        t_fine = np.arange(0, cycle_duration, target_dt)

        hip_fine  = f_hip(t_fine)
        knee_fine = f_knee(t_fine)

    elif mode == "positions":
        raise NotImplementedError(
            "positions 모드는 아직 미구현입니다. "
            "config.CSV_COLUMN_MAP['mode'] = 'angles' 로 설정하세요."
        )
    else:
        raise ValueError(f"Unknown CSV mode: {mode!r}")

    # n_cycles 반복
    t_out   = np.concatenate([t_fine + i * cycle_duration for i in range(n_cycles)])
    hip_out  = np.tile(hip_fine,  n_cycles)
    knee_out = np.tile(knee_fine, n_cycles)

    return GaitTrajectory(
        time=t_out,
        hip_angle=np.deg2rad(hip_out),
        knee_angle=np.deg2rad(knee_out),
        cycle_duration=cycle_duration,
        source_cycle_duration=source_cycle_duration,
        temporal_scale=temporal_scale,
    )
