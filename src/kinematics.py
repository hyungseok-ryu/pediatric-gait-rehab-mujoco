"""
kinematics.py
=============
사람 하지 FK와 보행 보조 로봇의 운동학 모듈.

좌표계 (project.md §2):
  X : 보행 진행 방향
  Z : 상향 (중력 반대)
  Y : 우측 (오른손 좌표계)

사람 하지:
  - Hip 관절: Global frame 원점에 고정.
  - Hip joint: Y축 회전, 양수 = flexion (다리가 앞으로).
  - Knee joint: Y축 회전, 양수 = flexion (다리가 뒤로 굽힘).
  - 운동은 X-Z 평면(시상면) 내에서만 발생.

보조 로봇:
  - exoskeleton은 사람 hip/knee 목표각을 직접 사용하므로 별도 IK가 없다.
  - open_chain은 독립 직렬 2-link 팔의 끝점을 발목 위치에 맞춘다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# 사람 하지 FK
# ─────────────────────────────────────────────────────────────────────────────

def human_leg_fk(
    hip_angle: float,
    knee_angle: float,
    thigh_length: float,
    shank_length: float,
    hip_pos: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    2-link 하지 정기구학 (FK). X-Z 평면.

    좌표 규약:
      - hip_angle  > 0 : flexion (대퇴가 앞 방향 +X 쪽으로 기울어짐)
      - knee_angle > 0 : flexion (하퇴가 뒤쪽으로 굽힘 → 하퇴 각도는 thigh 각도 - knee_angle)

    각도 → 분절 절대각 변환:
      θ_thigh   = hip_angle                 (Global Z로부터 반시계 = Y축 양방향 회전)
      θ_shank   = hip_angle - knee_angle     (knee flexion 은 thigh 각을 감소시킴)

    Parameters
    ----------
    hip_angle, knee_angle : float  [rad]
    thigh_length, shank_length : float [m]
    hip_pos : (3,) array or None  [m]. None → (0, 0, 0).

    Returns
    -------
    hip_pos, knee_pos, ankle_pos : (3,) arrays [m]
    """
    if hip_pos is None:
        hip_pos = np.zeros(3)

    # 분절 절대각(Global Y축 기준 회전각)
    # MuJoCo/biomechanics 관례: 하향(−Z) 방향이 neutral, Y축 회전 양수 = 앞(+X)
    theta_thigh = hip_angle          # thigh 분절 절대각 [rad]
    theta_shank = hip_angle - knee_angle  # shank 분절 절대각 [rad]

    # 관절 위치 계산 (X-Z 평면)
    knee_pos = hip_pos + thigh_length * np.array([
        np.sin(theta_thigh),   # x
        0.0,                   # y
        -np.cos(theta_thigh),  # z  (아래방향 = -Z)
    ])

    ankle_pos = knee_pos + shank_length * np.array([
        np.sin(theta_shank),
        0.0,
        -np.cos(theta_shank),
    ])

    return hip_pos.copy(), knee_pos, ankle_pos


def open_chain_ik(
    target_pos: NDArray[np.float64],
    base_pos: NDArray[np.float64],
    link1_length: float,
    link2_length: float,
    elbow_behind: bool = False,
) -> tuple[float, float] | None:
    """발목 목표 위치에 대한 시상면 2-link open-chain IK를 계산한다.

    반환값은 ``(motor1, motor2)``이며 motor2는 link1에 대한 상대각이다.
    ``elbow_behind=False``는 일반 exoskeleton과 같은 굽힘 분기를,
    ``True``는 중간 관절이 뒤쪽에 놓이는 반대 분기를 선택한다.
    """
    if link1_length <= 0 or link2_length <= 0:
        return None

    dx = float(target_pos[0] - base_pos[0])
    dz = float(target_pos[2] - base_pos[2])
    distance = float(np.hypot(dx, dz))
    if (
        distance < 1e-9
        or distance > link1_length + link2_length
        or distance < abs(link1_length - link2_length)
    ):
        return None

    cos_elbow = (
        distance**2 - link1_length**2 - link2_length**2
    ) / (2.0 * link1_length * link2_length)
    cos_elbow = float(np.clip(cos_elbow, -1.0, 1.0))
    elbow = np.arccos(cos_elbow)
    if not elbow_behind:
        elbow = -elbow

    target_angle = np.arctan2(dx, -dz)
    offset = np.arctan2(
        link2_length * np.sin(elbow),
        link1_length + link2_length * np.cos(elbow),
    )
    motor1 = target_angle - offset
    motor2 = elbow
    return float(motor1), float(motor2)


# ─────────────────────────────────────────────────────────────────────────────
# remote_parallelogram IK
# ─────────────────────────────────────────────────────────────────────────────

def robot_arm_ik(
    target_pos: NDArray[np.float64],
    base_pos:   NDArray[np.float64],
    link1_length: float,
    link2_length: float,
    elbow_down: bool = True,
    d_motor: float = 0.0,
    robot_type: str = "remote_parallelogram",
    motor_distance: float = 0.10,
    motor2_offset_z: float = 0.0,
    return_passive: bool = False,
    # ── remote_parallelogram 전용 기하 (config.RobotArmParams와 동일) ──
    remote_c_top_link: float = 0.08,
    remote_d_upper_link: float = 0.06,
    remote_e_motor1_joint_offset: float = 0.05,
    distal_standoff: float = 0.040,
) -> tuple[float, float, float] | tuple[float, float, float, float, float, float, float, float] | None:
    """
    remote_parallelogram의 역기구학(IK)을 계산한다.

    Parameters
    ----------
    target_pos   : (3,) [m]  end-effector 목표 위치
    base_pos     : (3,) [m]  로봇팔 base 위치
    link1_length : float [m]
    link2_length : float [m]
    elbow_down   : bool  True = elbow-down 선택
    d_motor      : float [m]  passive joint와 motor base 사이의 Z축 오프셋
    robot_type   : str  반드시 "remote_parallelogram"
    motor_distance: float [m]  motor1에서 motor2까지의 X 오프셋
    return_passive: bool
        True이면 폐루프 passive joint 초기각까지 포함한 8-tuple 반환

    Returns
    -------
    기본: (theta_passive, theta_motor1, theta_motor2) [rad]
    return_passive=True이면 passive/constraint 초기화용 8-tuple을 반환한다.
    """
    if robot_type != "remote_parallelogram":
        raise ValueError(
            "robot_arm_ik is only defined for robot_type='remote_parallelogram'"
        )

    dx = target_pos[0] - base_pos[0]
    dy = target_pos[1] - base_pos[1]
    dz = target_pos[2] - base_pos[2]

    # passive joint 각도 계산 (X축 회전)
    theta_passive = np.arctan2(dy, -dz)

    # passive joint에 의해 회전된 로컬 평면에서의 dz 성분 (음수 값)
    dz_local = -np.sqrt(dy**2 + dz**2)
    dz_local_motor = dz_local + d_motor

    if robot_type == "remote_parallelogram":
        # ──────────────────────────────────────────────────────────────
        # Corrected remote parallelogram IK.
        #
        # Four-bar/parallelogram vertices:
        #   A : motor1_joint(active), proximal point of drive link A-D
        #   B : passive point on the A-B-E plate, proximal point of link B-C
        #   C : passive pin at the tip of the C standoff attached to link2
        #   D : passive pin at the tip of the D standoff attached to link2
        #   E : passive pin on the A-B-E plate connected to the motor2 crank-rocker
        #
        # Geometry in the sagittal local plane after base_passive rotation:
        #   A-B is a real short plate link along local -Z.
        #   D-C is the matching short spacing along local -Z on the link2 standoffs.
        #   ee target = D + [fixed capture offset b] + [end-effector extension L2]
        #             = L1·dir(θ1) + Lc·dir(θ_plate + δc)
        #   D is the distal pin closer to the end-effector target; C is behind it along local -Z.
        #
        # Therefore target-foot IK can still be solved as an equivalent 2R
        # problem for θ1 and θp, then θ2 is obtained from the motor2 crank-rocker.
        # ──────────────────────────────────────────────────────────────
        L1, L2 = link1_length, link2_length
        a = remote_e_motor1_joint_offset
        d_upper = remote_d_upper_link
        c_top = remote_c_top_link
        if d_upper < 0.5 * a:
            return None
        e_local_x = -np.sqrt(max(0.0, d_upper**2 - (0.5 * a) ** 2))
        e_local_z = -0.5 * a

        # Effective second link from D to bracket_end target.
        # link2 is the end-effector link. C/D are passive pins at the tips of two
        # short perpendicular standoffs fixed to link2.
        # Local distal-frame vector from D to the bracket_end/ee target:
        #   [b, +L2]
        # where b is the standoff length from D to the link2 centerline.
        # C is located from D along local -Z at distance a, mirroring the A-B plate link.
        b = distal_standoff
        local_x = b
        local_z = L2
        Lc = np.hypot(local_x, local_z)
        dc = np.arctan2(local_x, -local_z)

        tx, tz = dx, dz_local_motor
        d = np.sqrt(tx**2 + tz**2)
        if d > L1 + Lc or d < abs(L1 - Lc):
            return None

        cos_g = (d**2 - L1**2 - Lc**2) / (2.0 * L1 * Lc)
        cos_g = np.clip(cos_g, -1.0, 1.0)
        gamma = np.arccos(cos_g)
        if not elbow_down:
            gamma = -gamma

        alpha = np.arctan2(tx, -tz)
        beta = np.arctan2(Lc * np.sin(gamma), L1 + Lc * np.cos(gamma))
        theta1 = alpha - beta                 # drive_link absolute angle = motor1
        theta_eff = theta1 + gamma            # effective second-link absolute angle
        theta_plate = theta_eff - dc          # distal link D→E and plate B→A absolute angle

        # motor2 crank-rocker IK: M2=(motor_distance, motor2_offset_z), target E on plate.
        # A-B=e, A-E=d, B-E=d, so E is fixed by the d,d,e triangle.
        Cx = e_local_x * np.cos(theta_plate) - e_local_z * np.sin(theta_plate)
        Cz = e_local_x * np.sin(theta_plate) + e_local_z * np.cos(theta_plate)
        Px, Pz = Cx - motor_distance, Cz - motor2_offset_z
        D2 = np.hypot(Px, Pz)
        if D2 < 1e-9:
            return None

        k = (c_top**2 + D2**2 - c_top**2) / (2.0 * c_top)
        ratio = k / D2
        if abs(ratio) > 1.0:
            return None
        ratio = float(np.clip(ratio, -1.0, 1.0))
        omega = np.arctan2(-Pz, Px)
        theta2 = np.arcsin(ratio) - omega
        theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))

        if return_passive:
            # Tree mapping for the corrected XML:
            #   plate_joint     absolute θp relative to passive_bracket
            #   coupler_joint   makes A→E parallel to B→D
            #   distal_D_joint  makes distal link D→E parallel to plate B→A
            #   distal_C_joint  is the distal passive hinge at C, connected to the EE-side C standoff
            #   rod_joint       closes motor2 crank-rocker loop
            plate_j = theta_plate
            coupler_j = theta1 - theta_plate
            distal_D_j = theta_plate - theta1
            distal_C_j = theta_plate - theta1

            ctipx = motor_distance + c_top * np.sin(theta2)
            ctipz = motor2_offset_z - c_top * np.cos(theta2)
            rod_abs = np.arctan2(Cx - ctipx, -(Cz - ctipz))
            rod_j = rod_abs - theta2

            return (float(theta_passive), float(theta1), float(theta2),
                    float(plate_j), float(coupler_j), float(distal_D_j),
                    float(distal_C_j), float(rod_j))

        return float(theta_passive), float(theta1), float(theta2)


def remote_exoskeleton_ik(
    hip_angle: float,
    knee_angle: float,
    motor_distance: float,
    motor2_offset_z: float,
    crank_length: float,
    upper_plate_link: float,
    joint_spacing: float,
) -> tuple[float, float, float, float, float, float] | None:
    """
    Hip-knee 목표각을 remote_exoskeleton의 관절 초기값으로 변환한다.

    motor1은 hip과 동축으로 exo thigh를 직접 구동한다. Knee actuator인
    motor2는 hip 뒤쪽/위쪽 고정 프레임에 있고, 동일 길이 crank/rod와 A-B-E
    plate를 통해 knee의 상대 회전을 원격 전달한다.

    Returns
    -------
    (motor1, motor2, plate, coupler, exo_knee, rod) [rad]
    """
    if upper_plate_link < 0.5 * joint_spacing:
        return None
    if crank_length <= 0:
        return None

    theta1 = float(hip_angle)
    theta_knee = float(knee_angle)

    # Knee flexion이 증가하면 shank 절대각은 hip-knee가 된다.
    theta_plate = theta1 - theta_knee
    theta_coupler = theta_knee

    # A-B와 D-C는 thigh/shank 중심선에 직교하는 뒤쪽(-X) crossbar다.
    # E는 A-E=B-E인 이등변 삼각 plate의 꼭짓점이다.
    e_local_x = -0.5 * joint_spacing
    e_local_z = np.sqrt(
        max(0.0, upper_plate_link**2 - (0.5 * joint_spacing) ** 2)
    )

    ex = (
        e_local_x * np.cos(theta_plate)
        - e_local_z * np.sin(theta_plate)
    )
    ez = (
        e_local_x * np.sin(theta_plate)
        + e_local_z * np.cos(theta_plate)
    )

    # Motor2 crank tip과 plate E 사이를 같은 길이의 rod로 닫는다.
    px = ex - motor_distance
    pz = ez - motor2_offset_z
    distance = np.hypot(px, pz)
    ratio = distance / (2.0 * crank_length)
    if distance < 1e-9 or ratio > 1.0:
        return None

    ratio = float(np.clip(ratio, -1.0, 1.0))
    omega = np.arctan2(-pz, px)
    # 두 원의 교점 중 뒤쪽 motor2에서 crank가 아래로 향하는 조립 분기를
    # 사용한다. 전체 보행 범위에서 연속적이고 dead-center를 지나지 않는다.
    theta2 = np.arcsin(ratio) - omega
    theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))

    crank_tip_x = motor_distance + crank_length * np.sin(theta2)
    crank_tip_z = motor2_offset_z - crank_length * np.cos(theta2)
    rod_abs = np.arctan2(ex - crank_tip_x, -(ez - crank_tip_z))
    rod_joint = np.arctan2(
        np.sin(rod_abs - theta2),
        np.cos(rod_abs - theta2),
    )

    return (
        theta1,
        float(theta2),
        float(theta_plate),
        float(theta_coupler),
        theta_knee,
        float(rod_joint),
    )


def remote_ankle_arm_ik(
    target_pos: NDArray[np.float64],
    base_pos: NDArray[np.float64],
    link1_length: float,
    link2_length: float,
    motor_distance: float,
    motor2_offset_z: float,
    crank_length: float,
    upper_plate_link: float,
    joint_spacing: float,
    knee_behind: bool = True,
) -> tuple[float, float, float, float, float, float] | None:
    """
    발목 위치를 추종하는 remote_ankle_arm의 폐루프 관절각을 계산한다.

    A-D/B-C는 길이 ``link1_length``인 평행 링크이고, D에서 발목 cuff까지
    ``link2_length``인 distal 링크가 이어진다. A-B와 D-C는 뒤쪽(-X)
    crossbar이며, motor2는 상부 crank/rod를 통해 plate angle을 구동한다.
    ``knee_behind=True``이면 중간 관절 D가 사람 뒤쪽에 놓이는
    4족 보행 로봇형 조립 분기를 선택한다.

    Returns
    -------
    (motor1, motor2, plate, coupler, distal, rod) [rad]
    """
    if link1_length <= 0 or link2_length <= 0 or crank_length <= 0:
        return None
    if upper_plate_link < 0.5 * joint_spacing:
        return None

    dx = float(target_pos[0] - base_pos[0])
    dz = float(target_pos[2] - base_pos[2])
    distance = np.hypot(dx, dz)
    if (
        distance > link1_length + link2_length
        or distance < abs(link1_length - link2_length)
        or distance < 1e-9
    ):
        return None

    cos_gamma = (
        distance**2 - link1_length**2 - link2_length**2
    ) / (2.0 * link1_length * link2_length)
    cos_gamma = float(np.clip(cos_gamma, -1.0, 1.0))
    gamma = np.arccos(cos_gamma)
    if not knee_behind:
        gamma = -gamma

    target_angle = np.arctan2(dx, -dz)
    beta = np.arctan2(
        link2_length * np.sin(gamma),
        link1_length + link2_length * np.cos(gamma),
    )
    theta1 = target_angle - beta
    theta_plate = theta1 + gamma
    theta_coupler = theta1 - theta_plate
    theta_distal = theta_plate - theta1

    # remote_exoskeleton과 동일한 posterior A-B-E plate 및 crank/rod.
    e_local_x = -0.5 * joint_spacing
    e_local_z = np.sqrt(
        max(0.0, upper_plate_link**2 - (0.5 * joint_spacing) ** 2)
    )
    ex = (
        e_local_x * np.cos(theta_plate)
        - e_local_z * np.sin(theta_plate)
    )
    ez = (
        e_local_x * np.sin(theta_plate)
        + e_local_z * np.cos(theta_plate)
    )

    px = ex - motor_distance
    pz = ez - motor2_offset_z
    crank_target_distance = np.hypot(px, pz)
    ratio = crank_target_distance / (2.0 * crank_length)
    if crank_target_distance < 1e-9 or ratio > 1.0:
        return None

    ratio = float(np.clip(ratio, -1.0, 1.0))
    omega = np.arctan2(-pz, px)
    theta2 = np.arcsin(ratio) - omega
    theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))

    crank_tip_x = motor_distance + crank_length * np.sin(theta2)
    crank_tip_z = motor2_offset_z - crank_length * np.cos(theta2)
    rod_abs = np.arctan2(ex - crank_tip_x, -(ez - crank_tip_z))
    rod_joint = np.arctan2(
        np.sin(rod_abs - theta2),
        np.cos(rod_abs - theta2),
    )

    return (
        float(theta1),
        float(theta2),
        float(theta_plate),
        float(theta_coupler),
        float(theta_distal),
        float(rod_joint),
    )


def compute_ankle_trajectory(
    hip_angles:  NDArray[np.float64],
    knee_angles: NDArray[np.float64],
    thigh_length: float,
    shank_length: float,
    hip_pos:      NDArray[np.float64] | None = None,
    scale_factor: float = 1.0,
) -> NDArray[np.float64]:
    """
    보행 데이터 → ankle 위치 시계열 계산.

    Parameters
    ----------
    hip_angles, knee_angles : (N,) [rad]
    thigh_length, shank_length : float [m]
    hip_pos : (3,) [m]  None → 원점
    scale_factor : float  ankle 궤적 선형 스케일링 비율 (§6.2)

    Returns
    -------
    ankle_traj : (N, 3) [m]
    """
    if hip_pos is None:
        hip_pos = np.zeros(3)

    N = len(hip_angles)
    ankle_traj = np.zeros((N, 3))

    for i in range(N):
        _, _, ankle = human_leg_fk(
            hip_angles[i], knee_angles[i],
            thigh_length, shank_length,
            hip_pos,
        )
        ankle_traj[i] = ankle

    if scale_factor != 1.0:
        # 중심(hip) 기준 스케일링
        center = hip_pos.copy()
        ankle_traj = center + scale_factor * (ankle_traj - center)

    return ankle_traj
