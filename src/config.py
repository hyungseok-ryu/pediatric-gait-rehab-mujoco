"""
config.py
=========
모든 설정(환자 파라미터, 보행 보조 로봇, 제어 게인, 경로 등)을 한 곳에 모은 모듈.
시뮬레이션 실행 전 이 파일의 값만 수정하면 전체 파이프라인에 반영된다.

Assumptions (project.md §11 참조):
- 기본 보행 CSV: 3–4세 Comfortable 오른쪽 hip/knee 평균 1주기.
- 시간축: CSV의 time 열을 우선 사용한다. 비교 시뮬레이션은 관절각 파형을
  유지하면서 CLI --cycle-duration으로 1주기 시간을 조정할 수 있다.
- exoskeleton 링크 길이: 대상자의 thigh/shank에 자동 정렬.
- open_chain 링크 길이: RobotArmParams 또는 CLI --link1/--link2로 결정.
- actuator와 구조 링크 질량은 각각 분리해 모델링한다.
- 발 질량: foot_mass(kg) 를 ankle 추종 부하에 포함.
- 토크 포화: torque_limit(Nm), None이면 미적용.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    # 기본 입력: 공개 3–4세 소아 4명의 Comfortable/Right hip·knee 평균.
    # 기존 속도별 입력은 legacy_gait_data_csv로 보존한다.
    "gait_data_csv": ROOT / "data" / "gait_3to4_comfortable_right_mean.csv",
    "legacy_gait_data_csv": ROOT / "data" / "gait_data.csv",
    "xml_model":     ROOT / "models" / "leg_and_arm.xml",
    "output_dir":    ROOT / "outputs",
    "plot_dir":      ROOT / "outputs" / "plots",
}


# ── CSV 컬럼 매핑 ──────────────────────────────────────────────────────────────
# (a) 관절각 모드: hip_angle, knee_angle (단위: degrees)
# (b) 위치 모드:   hip_x, hip_z, knee_x, knee_z, ankle_x, ankle_z (단위: m)
# 아래에서 mode = "angles" | "positions" 으로 선택
CSV_COLUMN_MAP = {
    "mode":       "angles",          # "angles" | "positions"
    "time":       "time",            # 없으면 Phase 또는 행 인덱스로 시간축 생성
    "hip_angle":  "hip_angle",       # [deg] 양수 = flexion
    "knee_angle": "knee_angle",      # [deg] 양수 = flexion
    # --- positions 모드 시 사용 ---
    "hip_x":    None,
    "hip_z":    None,
    "knee_x":   None,
    "knee_z":   None,
    "ankle_x":  None,
    "ankle_z":  None,
}

# CSV 샘플링 주기 (s). time 컬럼이 없을 때 사용.
GAIT_DATA_DT: float = 0.01   # 100 Hz 가정 (1 cycle ≈ 1.02 s for 102 samples)
# 기본 공개 파형의 stride time(0.7775 s)은 보존하되, 만 2세 시뮬레이션은
# 과도하게 빠른 재생을 피하도록 1주기를 1.0 s로 시간 확장한다.
DEFAULT_GAIT_CYCLE_DURATION: float = 1.0


# ── 한국 소아 만 1–2세 대표 체형 ─────────────────────────────────────────────
@dataclass(frozen=True)
class AnthropometricCase:
    """성장도표에서 선택한 시뮬레이션 대표 체형."""

    name: str
    age_months: int
    sex_basis: str
    percentile: str
    height: float  # [m]
    weight: float  # [kg]


# 2017 한국 소아청소년 성장도표는 3세 미만에 WHO 성장기준을 적용한다.
# "가장 작음/큼"은 관측된 절대 극단값이 아니라 재현 가능한 설계 경계인
# 3/97백분위수로 정의한다. 기본 평균 체형은 국내 KIPGroS의 22–23개월
# 남아·여아 50백분위수 평균(누운키 85.55 cm, 체중 11.95 kg)을 반올림했다.
KOREAN_TODDLER_CASES: tuple[AnthropometricCase, ...] = (
    AnthropometricCase(
        name="KR_1to2_small_P3",
        age_months=12,
        sex_basis="female",
        percentile="P3",
        height=0.692,
        weight=7.1,
    ),
    AnthropometricCase(
        name="KR_age2_KIPGroS_P50",
        age_months=23,
        sex_basis="female/male mean",
        percentile="P50",
        height=0.856,
        weight=12.0,
    ),
    AnthropometricCase(
        name="KR_1to2_large_P97",
        age_months=35,
        sex_basis="male",
        percentile="P97",
        height=1.023,
        weight=17.8,
    ),
)
DEFAULT_TODDLER_CASE = KOREAN_TODDLER_CASES[1]


# ── 환자(대상자) 파라미터 ─────────────────────────────────────────────────────
@dataclass
class SubjectParams:
    """
    대상자 신체 파라미터.
    None 으로 두면 anthropometry.py 의 Winter 비율로 자동 추정.
    """
    subject_height: float = DEFAULT_TODDLER_CASE.height  # [m] KIPGroS 22–23개월 P50 평균
    subject_weight: float = DEFAULT_TODDLER_CASE.weight  # [kg] KIPGroS 22–23개월 P50 평균
    thigh_length:   float | None = None   # [m] 대퇴 길이
    shank_length:   float | None = None   # [m] 하퇴 길이
    thigh_mass:     float | None = None   # [kg]
    shank_mass:     float | None = None   # [kg]
    foot_mass:      float | None = None  # [kg] None → 몸무게의 1.45%로 자동 추정
    # 좌우 hip joint center 사이 거리. None이면 키의 16%로 추정한다.
    hip_joint_spacing: float | None = None  # [m]

    # 보행 데이터 스케일링 옵션
    scale_trajectory: bool = True   # True → 대상자 다리 길이 기준으로 ankle 궤적 스케일링

SUBJECT = SubjectParams()

# 별도의 골반 계측값이 없을 때 사용하는 좌우 hip joint center 간격 비율.
# 시상면 단일 데이터로 양측 모델을 만들기 위한 기하 가정이며, 대상자별
# 골반 폭을 알고 있으면 SubjectParams.hip_joint_spacing으로 덮어쓴다.
HIP_JOINT_SPACING_HEIGHT_RATIO: float = 0.16


# ── 보행 보조 로봇 파라미터 ──────────────────────────────────────────────────
@dataclass
class RobotArmParams:
    """
    보행 보조 로봇 구조 파라미터.

    exoskeleton:
    - 사람의 hip/knee와 동축인 두 actuator를 사용한다.
    - 외골격 대퇴/하퇴 길이는 사람 분절 길이에 자동으로 맞춘다.

    open_chain:
    - hip 위치에 고정된 독립 2-link 직렬 로봇팔이다.
    - 사람 무릎과 결합하지 않고 end-effector cuff 위치만 발목에 연결한다.
    """
    robot_type:         str   = "exoskeleton"
    arm_link_mass:      float = 0.6    # [kg] legacy 모델 호환용(활성 모델은 미사용)
    exo_lateral_offset: float = 0.08   # [m] 사람 다리 중심선에서 외골격 링크까지 Y 간격

    # exoskeleton/open_chain 공통 질량
    # motor: CubeMars AK70-10급 통합 actuator(모터/감속기/드라이버 포함)
    exo_motor_mass: float = 0.521      # [kg] actuator 1개
    exo_link_mass:  float = 0.250      # [kg] thigh/shank 구조 링크·cuff 각 1개

    # open_chain: 사람 thigh/shank와 독립적인 직렬 로봇팔
    # 기본값은 대상자 분절 길이 × (1 + reach_margin)으로 자동 산정한다.
    # auto_size=False 또는 CLI --link1/--link2 지정 시 아래 수동값을 사용한다.
    open_chain_auto_size: bool = True
    open_chain_reach_margin: float = 0.10
    open_chain_link1_length: float = 0.231  # [m] 수동 fallback (만 2세 근사)
    open_chain_link2_length: float = 0.232  # [m] 수동 fallback (만 2세 근사)
    # 발목 cuff까지 도달하는 두 IK 해 중, 두 번째 motor/elbow가 사람
    # 무릎보다 뒤쪽(-X)에 놓이는 분기를 기본으로 사용한다.
    open_chain_elbow_behind: bool = True

    # 아래 remote_* 항목은 이전 결과 재현용 legacy 파라미터이며
    # 현재 활성 타입(exoskeleton/open_chain)에서는 사용하지 않는다.
    # remote_exoskeleton: hip 기준 상부 motor2와 crank/plate 전달부
    remote_exo_motor_dx:      float = -0.20 # [m] hip motor에서 knee motor까지 뒤쪽(-X)
    remote_exo_motor_dz:      float = 0.08  # [m] hip motor에서 knee motor까지 +Z
    remote_exo_crank_length:  float = 0.12  # [m] motor2 crank와 rod 공통 길이
    remote_exo_upper_link:    float = 0.08  # [m] A-E/B-E 삼각 plate 변 길이
    remote_exo_joint_spacing: float = 0.11  # [m] A-B 및 D-C 간격
    remote_exo_plate_mass:    float = 0.120 # [kg] A-B-E plate와 upper crossbar
    remote_exo_coupler_mass:  float = 0.180 # [kg] B-C 장축 링크
    remote_exo_crank_mass:    float = 0.080 # [kg] motor2 crank
    remote_exo_rod_mass:      float = 0.080 # [kg] crank-E 연결 rod

    # remote_ankle_arm: 사람 분절과 독립적인 로봇팔 링크 길이
    remote_ankle_link1_length: float = 0.35 # [m] A-D/B-C 평행 링크
    remote_ankle_link2_length: float = 0.35 # [m] D에서 발목 cuff까지 distal 링크

    # remote_parallelogram 공통/기존 파라미터
    link1_length:       float = 0.40   # [m] 구동 링크 길이
    link2_length:       float = 0.40   # [m] distal 링크 길이
    base_above_knee:    float = 0.10   # [m] base 위치를 무릎 관절 위로 올리는 거리
    x_offset:           float = -0.05  # [m] base X 위치
    y_offset:           float = 0.15   # [m] base Y 위치
    motor_offset_z:     float = 0.05   # [m] passive joint로부터 motor1까지 Z축 아래로 떨어진 오프셋 거리
    motor_distance:     float = 0.05   # [m] 두 모터(M1=hip, M2=crank) 축 사이 간격
    motor2_offset_z:    float = 0.05   # [m] Motor 1 대비 Motor 2의 수직 방향 추가 오프셋 (위로 양수)

    # ── remote_parallelogram 전용 기하 파라미터 ─────────────────────────────
    # 모두 "측정 전, 최적화 대상" 설계 변수. 의미는 kinematics/build_xml 주석 참조.
    # 사용자 스케치의 a,b,c,d,e,f,g 정의:
    #   remote_a_motor_dx           : Motor1 중심에서 Motor2 중심까지 수평 거리
    #   remote_b_motor_dz           : Motor1 중심에서 Motor2 중심까지 수직 거리
    #   remote_c_top_link           : 위쪽 motor2 crank/rod 두 링크의 공통 길이
    #   remote_d_upper_link         : passive plate 위쪽 B→E 링크 길이
    #   remote_e_motor1_joint_offset: Motor1 중심에서 B/C/D joint line까지의 수평/법선 오프셋
    #   remote_f_drive_link         : A-D 및 B-C 평행사변형 긴 링크 길이
    #   remote_g_distal_link        : distal/foot bar 길이
    remote_a_motor_dx:            float = 0.11
    remote_b_motor_dz:            float = 0.02
    remote_c_top_link:            float = 0.09
    remote_d_upper_link:          float = 0.08
    remote_e_motor1_joint_offset: float = 0.11
    remote_f_drive_link:          float = 0.30
    remote_g_distal_link:         float = 0.20

    rod_layer_offset:   float = 0.018  # [m] rod 적층 offset. motor1 rod=-Y(안쪽), coupler rod=+Y(바깥쪽)
    distal_standoff:   float = 0.040  # [m] EE link에서 C/D passive joint까지 올라가는 고정 standoff 길이
    cuff_length:        float = 0.10   # [m] 로봇팔 끝(EE)에서 발목까지 연결되는 커프의 고정 길이
    torque_limit:       float | None = None    # [Nm] None → 포화 없음 (모터 사이징 목적). 토크 포화 적용 시 30~100 범위로 설정.
    passive_damping:    float = 0.1    # [Nm·s/rad] passive joint 감쇠 계수
    control_mode:       str   = "robot_drives_human" # "ik_pd", "constraint_id", "robot_drives_human" (로봇팔 주도 보행)

ROBOT_ARM = RobotArmParams()


# ── 소아용 수동보행기 파라미터 ──────────────────────────────────────────────
@dataclass
class WalkerParams:
    """로봇을 지지하는 후방형 소아 수동보행기 외형 파라미터."""

    enabled: bool = True
    # None이면 대상자 키에 따라 안전 범위 안에서 자동 산정한다.
    frame_width: float | None = None   # [m] 좌우 바깥 프레임 중심 간격
    frame_length: float | None = None  # [m] 앞뒤 캐스터 축간 대략 길이
    handle_height_above_hip: float = 0.14  # [m]
    tube_radius: float = 0.012             # [m]
    caster_radius: float = 0.030           # [m]


WALKER = WalkerParams()


# ── PD 제어 게인 ──────────────────────────────────────────────────────────────
@dataclass
class PDGains:
    """
    모터별 PD 게인.
    τ = Kp*(θ_des - θ) + Kd*(θ̇_des - θ̇)
    """
    # 만 1–2세 세 대표 체형에서 공통으로 안정성을 확인한 기본값.
    Kp_motor1: float = 1500.0  # [Nm/rad]
    Kd_motor1: float = 3.0     # [Nm·s/rad]
    Kp_motor2: float = 1500.0  # [Nm/rad]
    Kd_motor2: float = 3.0     # [Nm·s/rad]

PD_GAINS = PDGains()


# ── 시뮬레이션 설정 ────────────────────────────────────────────────────────────
SIM_TIMESTEP:   float = 0.001   # [s] MuJoCo timestep (1000 Hz)
SIM_N_CYCLES:   int   = 5      # 보행 사이클 반복 횟수
RENDER_MODE:    str   = "viewer"  # "viewer" | "offscreen" | "none"
VIDEO_FPS:      int   = 60
VIDEO_PATH:     Path  = PATHS["output_dir"] / "simulation.mp4"
