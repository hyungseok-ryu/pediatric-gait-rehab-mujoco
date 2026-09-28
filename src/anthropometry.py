"""
anthropometry.py
================
Winter(1990) 인체측정학 비율을 사용한 신체 분절 파라미터 추정.

Reference:
    Winter, D. A. (1990). Biomechanics and Motor Control of Human Movement.
    Table 4.1 – Body Segment Parameters (mean proportions).
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentParams:
    """계산된 신체 분절 파라미터."""
    thigh_length: float   # [m]
    shank_length: float   # [m]
    thigh_mass:   float   # [kg]
    shank_mass:   float   # [kg]
    foot_mass:    float   # [kg]
    # 근위 끝(관절)으로부터 질량 중심까지의 거리 비율 (Winter Table 4.1)
    thigh_com_ratio: float = 0.433   # thigh_length 기준
    shank_com_ratio: float = 0.433   # shank_length 기준
    # 관성 모멘트 반경 비율 (k/L, Winter Table 4.1)
    thigh_rg_ratio:  float = 0.323
    shank_rg_ratio:  float = 0.302


def estimate_segment_params(
    height: float,
    weight: float,
    thigh_length: float | None = None,
    shank_length: float | None = None,
    thigh_mass:   float | None = None,
    shank_mass:   float | None = None,
    foot_mass:    float | None = None,
) -> SegmentParams:
    """
    Winter(1990) 비율로 신체 분절 파라미터를 추정한다.
    각 파라미터가 명시적으로 주어지면 그 값을 그대로 사용한다.

    Winter 비율 (Table 4.1):
        thigh_length ≈ 0.245 × height
        shank_length ≈ 0.246 × height
        foot_length  ≈ 0.152 × height  (참고용, 미사용)
        thigh_mass   ≈ 0.100 × body_mass
        shank_mass   ≈ 0.0465 × body_mass
        foot_mass    ≈ 0.0145 × body_mass

    Parameters
    ----------
    height : float
        대상자 키 [m].
    weight : float
        대상자 체중 [kg].
    thigh_length, shank_length, thigh_mass, shank_mass, foot_mass : float | None
        직접 지정할 경우 해당 값 사용, None 이면 Winter 비율로 추정.

    Returns
    -------
    SegmentParams
    """
    tl = thigh_length if thigh_length is not None else 0.245 * height
    sl = shank_length if shank_length is not None else 0.246 * height
    tm = thigh_mass   if thigh_mass   is not None else 0.100  * weight
    sm = shank_mass   if shank_mass   is not None else 0.0465 * weight
    fm = foot_mass    if foot_mass    is not None else 0.0145 * weight

    return SegmentParams(
        thigh_length=tl,
        shank_length=sl,
        thigh_mass=tm,
        shank_mass=sm,
        foot_mass=fm,
    )


def estimate_open_chain_link_lengths(
    thigh_length: float,
    shank_length: float,
    reach_margin: float = 0.10,
    resolution: float = 0.001,
) -> tuple[float, float]:
    """체형에 맞는 발목 추종 open-chain 링크 길이를 거칠게 산정한다.

    사람 대퇴·하퇴 길이에 동일한 도달 여유율을 적용한다. 기본 10% 여유는
    사람 다리가 완전히 신전되어도 로봇팔의 최대 도달거리 안에 들어오게 하며,
    2-link 로봇팔이 완전 신전 특이점에 놓이는 것을 피하기 위한 설계 여유다.
    최적 설계값이 아니라 체형별 초기 설계값으로 사용한다. 실제 제작에
    가까운 대략값이 되도록 각 링크를 ``resolution`` 단위로 올림한다.
    """
    if thigh_length <= 0 or shank_length <= 0:
        raise ValueError("thigh_length and shank_length must be positive")
    if reach_margin < 0:
        raise ValueError("reach_margin must be non-negative")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    scale = 1.0 + reach_margin
    link1 = np.ceil(scale * thigh_length / resolution) * resolution
    link2 = np.ceil(scale * shank_length / resolution) * resolution
    return float(link1), float(link2)


def compute_inertia_cylinder(mass: float, radius: float, length: float) -> tuple[float, float, float]:
    """
    균일 원통형 링크의 주축 관성 모멘트 (Ixx, Iyy, Izz) 반환.

    원통의 긴 축 = Z축 가정 (MuJoCo capsule 기본).
        Ixx = Iyy = (1/12)*m*L² + (1/4)*m*r²
        Izz = (1/2)*m*r²

    Parameters
    ----------
    mass   : [kg]
    radius : [m]
    length : [m]

    Returns
    -------
    (Ixx, Iyy, Izz) : tuple[float, float, float]  [kg·m²]
    """
    Ixx = (1 / 12) * mass * length ** 2 + (1 / 4) * mass * radius ** 2
    Iyy = Ixx
    Izz = 0.5 * mass * radius ** 2
    return Ixx, Iyy, Izz
