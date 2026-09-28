"""
build_xml.py
============
대상자 파라미터와 보행 보조 로봇 파라미터로부터 양측 MuJoCo XML을 생성한다.

XML 구조:
  worldbody
  ├── (조명 / 바닥 참조선)
  ├── body "hip" [오른쪽 사람 hip 관절]
  │   └── body "thigh"
  │       └── body "shank"
  │           └── site "ankle_site"
  ├── robot_type에 따른 오른쪽 보조 로봇
  └── 위 하지/로봇을 Y축 반사한 ``*_left`` 왼쪽 세트
      ├── exoskeleton: 사람 hip/knee와 동축인 2관절 외골격
      └── open_chain: 무릎 비결합형 직렬 2-link 발목 추종 로봇팔

사람 다리 제어:
  - hip_joint, knee_joint position actuator를 사용한다.
  - robot_drives_human에서는 사람 actuator gain을 0으로 두고 로봇이 구동한다.

보조 로봇 제어:
  - motor1, motor2는 외부 PD torque actuator이다.
  - exoskeleton의 motor1/motor2는 각각 hip/knee에 대응한다.
  - open_chain의 motor1/motor2는 발목 목표 위치의 2-link IK를 추종한다.
"""

from __future__ import annotations

import copy
import math
import textwrap
import xml.etree.ElementTree as ET


def _append_suffix(element: ET.Element, suffix: str) -> None:
    """복제된 MuJoCo 요소의 이름과 내부 참조에 접미사를 붙인다."""
    reference_attributes = {
        "joint",
        "joint1",
        "joint2",
        "site1",
        "site2",
        "objname",
        "tendon1",
        "tendon2",
        "body1",
        "body2",
    }
    for child in element.iter():
        if "name" in child.attrib:
            child.attrib["name"] += suffix
        for attribute in reference_attributes:
            if attribute in child.attrib:
                child.attrib[attribute] += suffix


def _reflect_y(element: ET.Element) -> None:
    """바디 트리를 Y=0 평면에 대해 반사한다(관절 각도 규약은 유지)."""
    for child in element.iter():
        for attribute in ("pos", "fromto"):
            value = child.attrib.get(attribute)
            if value is None:
                continue
            coordinates = [float(item) for item in value.split()]
            for index in range(1, len(coordinates), 3):
                coordinates[index] *= -1.0
            child.attrib[attribute] = " ".join(f"{item:.8g}" for item in coordinates)


def _shift_root_y(body: ET.Element, delta_y: float) -> None:
    coordinates = [float(item) for item in body.attrib.get("pos", "0 0 0").split()]
    coordinates[1] += delta_y
    body.attrib["pos"] = " ".join(f"{item:.8g}" for item in coordinates)


def _bilateralize_xml(
    xml_text: str,
    robot_type: str,
    hip_joint_spacing: float,
) -> str:
    """기존 오른쪽 하지/로봇을 보존하면서 반사된 왼쪽 세트를 추가한다.

    기존 무접미사 이름은 하위 분석 코드와의 호환을 위해 오른쪽에 사용한다.
    왼쪽의 body/joint/actuator/sensor/equality 이름에는 ``_left``가 붙는다.
    """
    if hip_joint_spacing <= 0:
        raise ValueError("hip_joint_spacing must be positive")

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(xml_text, parser=parser)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Generated MuJoCo XML has no worldbody")

    robot_root_names = {
        "exoskeleton": "exoskeleton_base",
        "open_chain": "open_chain_base",
        "remote_exoskeleton": "remote_exoskeleton_base",
        "remote_ankle_arm": "remote_ankle_base",
        "remote_parallelogram": "arm_base",
    }
    robot_root_name = robot_root_names[robot_type]
    right_hip = worldbody.find("./body[@name='hip']")
    right_robot = worldbody.find(f"./body[@name='{robot_root_name}']")
    if right_hip is None or right_robot is None:
        raise ValueError("Could not locate unilateral human/robot roots")

    # 기존 단일 모델은 Y=0의 다리와 -Y의 로봇이다. 오른쪽 다리를 골반
    # 중심에서 -spacing/2로 옮기고 로봇도 같은 양만큼 함께 옮긴다.
    half_spacing = 0.5 * hip_joint_spacing
    _shift_root_y(right_hip, -half_spacing)
    _shift_root_y(right_robot, -half_spacing)

    left_hip = copy.deepcopy(right_hip)
    _append_suffix(left_hip, "_left")
    _reflect_y(left_hip)
    worldbody.append(left_hip)

    left_robot = copy.deepcopy(right_robot)
    _append_suffix(left_robot, "_left")
    _reflect_y(left_robot)
    worldbody.append(left_robot)

    # 월드바디 밖에서 이름으로 관절/사이트를 참조하는 섹션도 좌측용으로
    # 한 번 더 복제한다. 활성 모델(exoskeleton/open_chain)에는 tendon이
    # 없지만 legacy 구조도 안전하게 생성할 수 있도록 함께 처리한다.
    for section_name in ("actuator", "sensor", "tendon", "equality"):
        section = root.find(section_name)
        if section is None:
            continue
        right_elements = list(section)
        for right_element in right_elements:
            left_element = copy.deepcopy(right_element)
            _append_suffix(left_element, "_left")
            section.append(left_element)

    root.attrib["model"] = f"{root.attrib.get('model', 'leg_and_arm')}_bilateral"
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root,
        encoding="unicode",
    ) + "\n"


def build_xml(
    thigh_length: float,
    shank_length: float,
    thigh_mass:   float,
    shank_mass:   float,
    foot_mass:    float,
    link1_length: float,
    link2_length: float,
    arm_link_mass: float,
    base_pos:     tuple[float, float, float],
    cuff_length:  float,
    passive_damping: float,
    control_mode: str,
    sim_timestep: float,
    motor_offset_z: float,
    robot_type: str = "exoskeleton",
    motor_distance: float = 0.10,
    motor2_offset_z: float = 0.0,
    exo_lateral_offset: float = 0.08,
    exo_motor_mass: float = 0.521,
    exo_link_mass: float = 0.250,
    # ── remote_exoskeleton 전용 상부 전달기구 ──
    remote_exo_motor_dx: float = -0.10,
    remote_exo_motor_dz: float = 0.08,
    remote_exo_crank_length: float = 0.12,
    remote_exo_upper_link: float = 0.08,
    remote_exo_joint_spacing: float = 0.11,
    remote_exo_plate_mass: float = 0.120,
    remote_exo_coupler_mass: float = 0.180,
    remote_exo_crank_mass: float = 0.080,
    remote_exo_rod_mass: float = 0.080,
    # ── remote_parallelogram 전용 기하 (config.RobotArmParams와 동일 의미) ──
    remote_c_top_link: float = 0.08,
    remote_d_upper_link: float = 0.06,
    remote_e_motor1_joint_offset: float = 0.05,
    rod_layer_offset: float = 0.018,
    distal_standoff: float = 0.030,
    hip_joint_spacing: float = 0.14,
    bilateral: bool = True,
    walker_enabled: bool = True,
    walker_frame_width: float = 0.58,
    walker_frame_length: float = 0.66,
    walker_handle_height_above_hip: float = 0.14,
    walker_tube_radius: float = 0.012,
    walker_caster_radius: float = 0.030,
) -> str:
    """
    MuJoCo XML 문자열을 생성하여 반환한다.

    Parameters
    ----------
    thigh_length, shank_length : [m]
    thigh_mass, shank_mass, foot_mass : [kg]
    link1_length, link2_length : [m]  remote_parallelogram 링크 길이
    arm_link_mass : [kg]  remote_parallelogram 주 링크 1개 질량
    exo_motor_mass : [kg]  exoskeleton 통합 actuator 1개 질량
    exo_link_mass : [kg]  exoskeleton thigh/shank 구조 링크 1개 질량
    base_pos : (x, y, z) [m]  보조 로봇 base 위치
    passive_damping : [Nm·s/rad]
    sim_timestep : [s]
    motor_offset_z: [m] passive joint부터 motor1까지 Z축 아래로 내려간 거리

    Returns
    -------
    xml_str : str
    """
    supported_robot_types = {
        "exoskeleton",
        "open_chain",
    }
    if robot_type not in supported_robot_types:
        raise ValueError(
            f"Unsupported robot_type {robot_type!r}; "
            f"expected one of {sorted(supported_robot_types)}"
        )
    if robot_type in {
        "exoskeleton",
        "open_chain",
        "remote_ankle_arm",
        "remote_exoskeleton",
    }:
        exo_masses = {
            "exo_motor_mass": exo_motor_mass,
            "exo_link_mass": exo_link_mass,
        }
        if robot_type in {"remote_ankle_arm", "remote_exoskeleton"}:
            exo_masses.update(
                remote_exo_plate_mass=remote_exo_plate_mass,
                remote_exo_coupler_mass=remote_exo_coupler_mass,
                remote_exo_crank_mass=remote_exo_crank_mass,
                remote_exo_rod_mass=remote_exo_rod_mass,
            )
        invalid = [name for name, mass in exo_masses.items() if mass <= 0]
        if invalid:
            raise ValueError(
                f"Exoskeleton masses must be positive: {', '.join(invalid)}"
            )

    if (
        robot_type == "remote_parallelogram"
        and remote_d_upper_link < 0.5 * remote_e_motor1_joint_offset
    ):
        raise ValueError(
            "remote_d_upper_link must be at least remote_e_motor1_joint_offset / 2 "
            "for the d,d,e isosceles triangle."
        )
    if (
        robot_type in {"remote_ankle_arm", "remote_exoskeleton"}
        and remote_exo_upper_link < 0.5 * remote_exo_joint_spacing
    ):
        raise ValueError(
            "remote_exo_upper_link must be at least "
            "remote_exo_joint_spacing / 2."
        )
    plate_e_local_x = -math.sqrt(
        max(0.0, remote_d_upper_link**2 - (0.5 * remote_e_motor1_joint_offset) ** 2)
    )
    plate_e_local_z = -0.5 * remote_e_motor1_joint_offset

    # 링크 반지름 (시각화용, 구조에는 영향 없음)
    r_leg  = 0.04   # [m] 다리 링크 반지름
    r_arm  = 0.025  # [m] 로봇팔 링크 반지름

    # 사람 다리 링크 관성 (균일 원통 가정)
    # Ixx = Iyy = (1/12)mL² + (1/4)mr²,  Izz = (1/2)mr²
    def inertia_str(mass: float, length: float, r: float) -> str:
        ixx = (1/12)*mass*length**2 + (1/4)*mass*r**2
        izz = 0.5*mass*r**2
        return f'{ixx:.6f} {ixx:.6f} {izz:.6f}'

    def cylinder_y_inertia_str(
        mass: float,
        radius: float,
        length: float,
    ) -> str:
        """Y축 원통 actuator의 중심 기준 대각 관성."""
        i_xz = mass * (3.0 * radius**2 + length**2) / 12.0
        i_y = 0.5 * mass * radius**2
        return f"{i_xz:.6f} {i_y:.6f} {i_xz:.6f}"

    def actuator_visual_xml(
        prefix: str,
        radius: float,
        half_width: float,
        indent: str,
    ) -> str:
        """단순 원통 대신 housing/flange/bolt가 보이는 actuator 외형을 만든다.

        반환되는 geom은 모두 시각화 전용이다. 각 motor body에 명시된 단일
        inertial이 질량과 관성을 담당하므로 이 외형은 동역학을 변경하지 않는다.
        고정 카메라가 -Y에 있어 output flange도 -Y 측에 배치한다.
        """
        housing_half = 0.76 * half_width
        cap_inner_y = housing_half
        flange_inner_y = -half_width
        flange_outer_y = -half_width - 0.005
        hub_outer_y = flange_outer_y - 0.007
        bolt_y = hub_outer_y - 0.001
        bolt_radius = 0.49 * radius
        bolt_geoms = []
        for index in range(6):
            angle = 2.0 * math.pi * index / 6.0
            bolt_x = bolt_radius * math.cos(angle)
            bolt_z = bolt_radius * math.sin(angle)
            bolt_geoms.append(
                f'<geom name="{prefix}_bolt_{index + 1}" type="sphere" '
                f'pos="{bolt_x:.4f} {bolt_y:.4f} {bolt_z:.4f}" size="0.0032" '
                'material="mat_motor_bolt" group="1" contype="0" conaffinity="0"/>'
            )

        geoms = [
            f'<geom name="{prefix}_housing" type="cylinder" '
            f'fromto="0 {-housing_half:.4f} 0 0 {housing_half:.4f} 0" '
            f'size="{radius:.4f}" material="mat_motor_housing" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_front_cap" type="cylinder" '
            f'fromto="0 {-half_width:.4f} 0 0 {-cap_inner_y:.4f} 0" '
            f'size="{0.965 * radius:.4f}" material="mat_motor_cap" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_rear_cap" type="cylinder" '
            f'fromto="0 {cap_inner_y:.4f} 0 0 {half_width:.4f} 0" '
            f'size="{0.965 * radius:.4f}" material="mat_motor_cap" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_center_band" type="cylinder" '
            'fromto="0 -0.0020 0 0 0.0020 0" '
            f'size="{1.018 * radius:.4f}" material="mat_motor_band" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_output_flange" type="cylinder" '
            f'fromto="0 {flange_outer_y:.4f} 0 0 {flange_inner_y:.4f} 0" '
            f'size="{0.72 * radius:.4f}" material="mat_motor_metal" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_output_hub" type="cylinder" '
            f'fromto="0 {hub_outer_y:.4f} 0 0 {flange_outer_y:.4f} 0" '
            f'size="{0.31 * radius:.4f}" material="mat_motor_hub" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_connector" type="box" '
            f'pos="{0.70 * radius:.4f} 0 {0.72 * radius:.4f}" '
            f'size="0.0080 {0.62 * half_width:.4f} 0.0070" '
            'material="mat_motor_connector" group="1" contype="0" conaffinity="0"/>',
            *bolt_geoms,
        ]
        return textwrap.indent("\n".join(geoms), indent)

    def structural_link_visual_xml(
        prefix: str,
        length: float,
        output_y: float,
        indent: str,
    ) -> str:
        """두께가 있는 link plate와 양단 체결부를 시각화한다."""
        end_clear = min(0.032, 0.18 * length)
        plate_half_length = 0.5 * (length - 2.0 * end_clear)
        plate_center_z = -0.5 * length
        accent_y = output_y - 0.0058
        distal_z = -length
        geoms = [
            f'<geom name="{prefix}_plate" type="box" '
            f'pos="0 {output_y:.4f} {plate_center_z:.4f}" '
            f'size="0.0130 0.0050 {plate_half_length:.4f}" '
            'material="mat_link_structure" group="1" contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_accent" type="box" '
            f'pos="0 {accent_y:.4f} {plate_center_z:.4f}" '
            f'size="0.0075 0.0012 {0.88 * plate_half_length:.4f}" '
            'material="mat_link_accent" group="1" contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_proximal_bracket" type="capsule" '
            f'fromto="0 {output_y:.4f} 0 0 {output_y:.4f} {-end_clear:.4f}" '
            'size="0.0110" material="mat_link_structure" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_distal_bracket" type="capsule" '
            f'fromto="0 {output_y:.4f} {-length + end_clear:.4f} '
            f'0 {output_y:.4f} {distal_z:.4f}" size="0.0110" '
            'material="mat_link_structure" group="1" contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_proximal_pin" type="cylinder" '
            f'fromto="0 {output_y - 0.007:.4f} 0 0 {output_y + 0.007:.4f} 0" '
            'size="0.0100" material="mat_link_pin" group="1" '
            'contype="0" conaffinity="0"/>',
            f'<geom name="{prefix}_distal_pin" type="cylinder" '
            f'fromto="0 {output_y - 0.007:.4f} {distal_z:.4f} '
            f'0 {output_y + 0.007:.4f} {distal_z:.4f}" size="0.0100" '
            'material="mat_link_pin" group="1" contype="0" conaffinity="0"/>',
        ]
        return textwrap.indent("\n".join(geoms), indent)

    def walker_visual_xml(floor_height: float) -> str:
        """고정식 후방형 소아 보행기와 로봇 장착 프레임을 만든다.

        바퀴를 포함한 모든 geom은 시각화 전용이며 collision/mass가 없다.
        따라서 보행기는 움직이지 않고 기존 공중 보행 동역학에도 영향을 주지
        않는다.
        """
        if not walker_enabled:
            return ""
        if min(
            walker_frame_width,
            walker_frame_length,
            walker_handle_height_above_hip,
            walker_tube_radius,
            walker_caster_radius,
        ) <= 0:
            raise ValueError("Walker dimensions must be positive")

        required_half_width = (
            0.5 * hip_joint_spacing + abs(exo_lateral_offset) + 0.10
        )
        half_width = max(0.5 * walker_frame_width, required_half_width)
        rear_x = -0.48 * walker_frame_length
        front_x = 0.52 * walker_frame_length
        post_x = rear_x + 0.065
        handle_z = walker_handle_height_above_hip
        wheel_z = floor_height + walker_caster_radius
        base_z = wheel_z + walker_caster_radius + 0.025
        crossbar_z = 0.025
        mount_x = -0.13
        robot_base_y = 0.5 * hip_joint_spacing + abs(exo_lateral_offset)
        tube_r = walker_tube_radius
        caster_r = walker_caster_radius
        geoms: list[str] = []

        def capsule(
            name: str,
            start: tuple[float, float, float],
            end: tuple[float, float, float],
            radius: float,
            material: str,
        ) -> None:
            geoms.append(
                f'<geom name="{name}" type="capsule" '
                f'fromto="{start[0]:.4f} {start[1]:.4f} {start[2]:.4f} '
                f'{end[0]:.4f} {end[1]:.4f} {end[2]:.4f}" '
                f'size="{radius:.4f}" material="{material}" group="2" '
                'contype="0" conaffinity="0"/>'
            )

        def cylinder(
            name: str,
            start: tuple[float, float, float],
            end: tuple[float, float, float],
            radius: float,
            material: str,
        ) -> None:
            geoms.append(
                f'<geom name="{name}" type="cylinder" '
                f'fromto="{start[0]:.4f} {start[1]:.4f} {start[2]:.4f} '
                f'{end[0]:.4f} {end[1]:.4f} {end[2]:.4f}" '
                f'size="{radius:.4f}" material="{material}" group="2" '
                'contype="0" conaffinity="0"/>'
            )

        def box(
            name: str,
            pos: tuple[float, float, float],
            size: tuple[float, float, float],
            material: str,
        ) -> None:
            geoms.append(
                f'<geom name="{name}" type="box" '
                f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
                f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" '
                f'material="{material}" group="2" '
                'contype="0" conaffinity="0"/>'
            )

        # 하부 U 프레임: 보행 방향 앞쪽은 다리가 자유롭게 지나가도록 열어 둔다.
        for side_name, side_sign in (("right", -1.0), ("left", 1.0)):
            y = side_sign * half_width
            capsule(
                f"walker_base_rail_{side_name}",
                (rear_x, y, base_z),
                (front_x, y, base_z),
                tube_r,
                "mat_walker_blue",
            )
            capsule(
                f"walker_front_inrigger_{side_name}",
                (front_x, y, base_z),
                (front_x, side_sign * (half_width - 0.085), base_z),
                tube_r,
                "mat_walker_blue",
            )
        capsule(
            "walker_rear_lower_crossbar",
            (rear_x, -half_width, base_z),
            (rear_x, half_width, base_z),
            tube_r,
            "mat_walker_blue",
        )

        # 좌우 텔레스코픽 지주, 상부 손잡이, 전완 패드.
        for side_name, side_sign in (("right", -1.0), ("left", 1.0)):
            y = side_sign * half_width
            capsule(
                f"walker_upright_{side_name}",
                (post_x, y, base_z),
                (post_x, y, handle_z),
                tube_r,
                "mat_walker_blue",
            )
            cylinder(
                f"walker_telescopic_insert_{side_name}",
                (post_x, y, -0.19),
                (post_x, y, -0.055),
                0.72 * tube_r,
                "mat_walker_metal",
            )
            box(
                f"walker_height_clamp_{side_name}",
                (post_x, y, -0.075),
                (0.020, 0.020, 0.016),
                "mat_walker_clamp",
            )
            capsule(
                f"walker_upper_rail_{side_name}",
                (post_x, y, handle_z),
                (0.225, y, handle_z + 0.035),
                tube_r,
                "mat_walker_blue",
            )
            capsule(
                f"walker_handle_grip_{side_name}",
                (0.115, y, handle_z + 0.025),
                (0.255, y, handle_z + 0.040),
                1.38 * tube_r,
                "mat_walker_grip",
            )
            capsule(
                f"walker_armrest_{side_name}",
                (-0.020, side_sign * (half_width - 0.030), handle_z + 0.058),
                (0.115, side_sign * (half_width - 0.030), handle_z + 0.058),
                0.034,
                "mat_walker_pad",
            )
            capsule(
                f"walker_armrest_support_{side_name}",
                (0.015, y, handle_z),
                (0.015, side_sign * (half_width - 0.030), handle_z + 0.035),
                0.008,
                "mat_walker_metal",
            )

        capsule(
            "walker_rear_upper_crossbar",
            (post_x, -half_width, handle_z),
            (post_x, half_width, handle_z),
            tube_r,
            "mat_walker_blue",
        )
        capsule(
            "walker_robot_crossbar",
            (mount_x, -half_width, crossbar_z),
            (mount_x, half_width, crossbar_z),
            1.12 * tube_r,
            "mat_walker_blue",
        )

        # 골반/등 지지대와 유연한 하부 스트랩의 외형.
        capsule(
            "walker_pelvic_back_pad",
            (
                post_x + 0.040,
                -min(0.135, half_width - 0.085),
                -0.025,
            ),
            (
                post_x + 0.040,
                min(0.135, half_width - 0.085),
                -0.025,
            ),
            0.043,
            "mat_walker_pad",
        )
        capsule(
            "walker_pelvic_sling",
            (-0.025, -min(0.145, half_width - 0.080), -0.115),
            (-0.025, min(0.145, half_width - 0.080), -0.115),
            0.023,
            "mat_walker_pad",
        )
        for side_name, side_sign in (("right", -1.0), ("left", 1.0)):
            pad_y = side_sign * (0.5 * hip_joint_spacing + 0.050)
            geoms.append(
                f'<geom name="walker_pelvic_side_pad_{side_name}" '
                f'type="ellipsoid" pos="-0.055 {pad_y:.4f} -0.030" '
                'size="0.044 0.030 0.060" material="mat_walker_pad" '
                'group="2" contype="0" conaffinity="0"/>'
            )

            # 프레임에서 각 로봇 hip base까지 이어지는 고정 장착 브래킷.
            robot_y = side_sign * robot_base_y
            capsule(
                f"walker_robot_mount_arm_{side_name}",
                (mount_x, robot_y, crossbar_z),
                (0.0, robot_y, 0.0),
                0.010,
                "mat_walker_metal",
            )
            box(
                f"walker_robot_mount_plate_{side_name}",
                (-0.020, robot_y, 0.0),
                (0.034, 0.010, 0.045),
                "mat_walker_clamp",
            )
            capsule(
                f"walker_diagonal_brace_{side_name}",
                (post_x, side_sign * half_width, -0.205),
                (mount_x, robot_y, crossbar_z),
                0.008,
                "mat_walker_blue",
            )

        # 네 캐스터는 외형만 갖는 고정 geom이다. 바퀴/포크/허브를 분리해
        # 실제 소형 캐스터처럼 보이게 하지만 joint와 contact는 두지 않는다.
        for x_name, frame_x, trail in (
            ("rear", rear_x, -0.018),
            ("front", front_x, 0.018),
        ):
            for side_name, side_sign in (("right", -1.0), ("left", 1.0)):
                y = side_sign * half_width
                wheel_x = frame_x + trail
                prefix = f"walker_caster_{x_name}_{side_name}"
                cylinder(
                    f"{prefix}_swivel",
                    (frame_x, y, base_z - 0.040),
                    (frame_x, y, base_z + 0.006),
                    0.017,
                    "mat_walker_clamp",
                )
                for fork_sign in (-1.0, 1.0):
                    fork_y = y + fork_sign * 0.022
                    capsule(
                        f"{prefix}_fork_{'a' if fork_sign < 0 else 'b'}",
                        (frame_x, fork_y, base_z - 0.025),
                        (wheel_x, fork_y, wheel_z),
                        0.006,
                        "mat_walker_metal",
                    )
                cylinder(
                    f"{prefix}_wheel",
                    (wheel_x, y - 0.024, wheel_z),
                    (wheel_x, y + 0.024, wheel_z),
                    caster_r,
                    "mat_walker_rubber",
                )
                cylinder(
                    f"{prefix}_hub",
                    (wheel_x, y - 0.026, wheel_z),
                    (wheel_x, y + 0.026, wheel_z),
                    0.010,
                    "mat_walker_hub",
                )

        return (
            '    <!-- 고정식 소아 수동보행기: 시각화 전용, 동역학/접촉 없음 -->\n'
            '    <body name="pediatric_walker" pos="0 0 0">\n'
            + textwrap.indent("\n".join(geoms), "      ")
            + "\n    </body>"
        )

    def loft_z_mesh_xml(
        name: str,
        sections: list[tuple[float, float, float, float]],
        radial_segments: int = 24,
    ) -> str:
        """(z, x-center, x-radius, y-radius) 단면을 잇는 limb mesh asset."""
        vertices: list[tuple[float, float, float]] = []
        for z_pos, x_center, x_radius, y_radius in sections:
            for index in range(radial_segments):
                angle = 2.0 * math.pi * index / radial_segments
                vertices.append(
                    (
                        x_center + x_radius * math.cos(angle),
                        y_radius * math.sin(angle),
                        z_pos,
                    )
                )

        faces: list[tuple[int, int, int]] = []
        for section_index in range(len(sections) - 1):
            start = section_index * radial_segments
            next_start = (section_index + 1) * radial_segments
            for index in range(radial_segments):
                following = (index + 1) % radial_segments
                a = start + index
                b = start + following
                c = next_start + index
                d = next_start + following
                faces.extend([(a, c, b), (b, c, d)])

        proximal_center = len(vertices)
        distal_center = proximal_center + 1
        first = sections[0]
        last = sections[-1]
        vertices.extend([(first[1], 0.0, first[0]), (last[1], 0.0, last[0])])
        last_start = (len(sections) - 1) * radial_segments
        for index in range(radial_segments):
            following = (index + 1) % radial_segments
            faces.append((proximal_center, following, index))
            faces.append((distal_center, last_start + index, last_start + following))

        vertex_text = " ".join(
            f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices
        )
        face_text = " ".join(f"{a} {b} {c}" for a, b, c in faces)
        return f'<mesh name="{name}" vertex="{vertex_text}" face="{face_text}"/>'

    def loft_x_mesh_xml(
        name: str,
        sections: list[tuple[float, float, float, float]],
        radial_segments: int = 24,
    ) -> str:
        """(x, z-center, y-radius, z-radius) 단면을 잇는 foot mesh asset."""
        vertices: list[tuple[float, float, float]] = []
        for x_pos, z_center, y_radius, z_radius in sections:
            for index in range(radial_segments):
                angle = 2.0 * math.pi * index / radial_segments
                vertices.append(
                    (
                        x_pos,
                        y_radius * math.sin(angle),
                        z_center + z_radius * math.cos(angle),
                    )
                )

        faces: list[tuple[int, int, int]] = []
        for section_index in range(len(sections) - 1):
            start = section_index * radial_segments
            next_start = (section_index + 1) * radial_segments
            for index in range(radial_segments):
                following = (index + 1) % radial_segments
                a = start + index
                b = start + following
                c = next_start + index
                d = next_start + following
                faces.extend([(a, b, c), (b, d, c)])

        heel_center = len(vertices)
        toe_center = heel_center + 1
        first = sections[0]
        last = sections[-1]
        vertices.extend([(first[0], 0.0, first[1]), (last[0], 0.0, last[1])])
        last_start = (len(sections) - 1) * radial_segments
        for index in range(radial_segments):
            following = (index + 1) % radial_segments
            faces.append((heel_center, index, following))
            faces.append((toe_center, last_start + following, last_start + index))

        vertex_text = " ".join(
            f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices
        )
        face_text = " ".join(f"{a} {b} {c}" for a, b, c in faces)
        return f'<mesh name="{name}" vertex="{vertex_text}" face="{face_text}"/>'

    def sleeve_z_mesh_xml(
        name: str,
        sections: list[tuple[float, float, float]],
        radial_segments: int = 32,
    ) -> str:
        """발목을 감싸는 개방형 타원 슬리브 mesh를 만든다.

        sections의 각 항목은 (z, x-radius, y-radius)이다. 상·하단을 막지
        않아 피부 위에 씌운 부드러운 패딩 커프처럼 보이게 한다.
        """
        vertices: list[tuple[float, float, float]] = []
        for z_pos, x_radius, y_radius in sections:
            for index in range(radial_segments):
                angle = 2.0 * math.pi * index / radial_segments
                vertices.append(
                    (
                        x_radius * math.cos(angle),
                        y_radius * math.sin(angle),
                        z_pos,
                    )
                )

        faces: list[tuple[int, int, int]] = []
        for section_index in range(len(sections) - 1):
            start = section_index * radial_segments
            next_start = (section_index + 1) * radial_segments
            for index in range(radial_segments):
                following = (index + 1) % radial_segments
                a = start + index
                b = start + following
                c = next_start + index
                d = next_start + following
                faces.extend([(a, c, b), (b, c, d)])

        vertex_text = " ".join(
            f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices
        )
        face_text = " ".join(f"{a} {b} {c}" for a, b, c in faces)
        return f'<mesh name="{name}" vertex="{vertex_text}" face="{face_text}"/>'

    thigh_inertia = inertia_str(thigh_mass, thigh_length, r_leg)
    shank_inertia = inertia_str(shank_mass, shank_length, r_leg)
    link1_mass = arm_link_mass
    link2_mass = arm_link_mass

    # remote_parallelogram: 전달부(plate/coupler/crank/rod)의 보조 질량 (보수적 소질량)
    aux_mass = max(0.02, arm_link_mass * 0.15)
    aux_inertia = "1e-5 1e-5 1e-5"

    link1_inertia = inertia_str(link1_mass, link1_length, r_arm)
    link2_inertia = inertia_str(link2_mass, link2_length, r_arm)

    # thigh CoM: 근위 끝에서 0.433*L 아래
    thigh_com_z = -0.433 * thigh_length
    shank_com_z = -0.433 * shank_length
    link1_com_z = -0.5   * link1_length
    link2_com_z = -0.5   * link2_length

    # 관절 위치 (parent body 기준)
    knee_pos_z = -thigh_length   # thigh body 기준 knee 관절
    ankle_rel_z = -shank_length  # shank body 기준 ankle site

    floor_z = -(thigh_length + shank_length + 0.1)
    walker_xml = walker_visual_xml(floor_z)

    # 연속 단면을 갖는 사람 하지 시각 mesh. 각 단면 반경은 소아 하지의
    # 완만한 taper와 종아리 윤곽만 표현하며 동역학 inertial에는 관여하지 않는다.
    human_thigh_mesh = loft_z_mesh_xml(
        "human_thigh_mesh",
        [
            (0.0, 0.000, 0.040, 0.037),
            (-0.08 * thigh_length, 0.001, 0.044, 0.040),
            (-0.32 * thigh_length, 0.004, 0.047, 0.041),
            (-0.58 * thigh_length, 0.003, 0.043, 0.038),
            (-0.82 * thigh_length, 0.001, 0.036, 0.033),
            (-thigh_length, 0.003, 0.033, 0.031),
        ],
    )
    human_shank_mesh = loft_z_mesh_xml(
        "human_shank_mesh",
        [
            (0.0, 0.003, 0.034, 0.031),
            (-0.12 * shank_length, 0.000, 0.033, 0.030),
            (-0.30 * shank_length, -0.007, 0.038, 0.034),
            (-0.50 * shank_length, -0.005, 0.034, 0.030),
            (-0.75 * shank_length, -0.001, 0.028, 0.026),
            (-shank_length, 0.000, 0.024, 0.023),
        ],
    )
    human_foot_mesh = loft_x_mesh_xml(
        "human_foot_mesh",
        [
            (-0.030, -0.010, 0.014, 0.018),
            (-0.015, -0.010, 0.028, 0.027),
            (0.025, -0.014, 0.035, 0.028),
            (0.075, -0.019, 0.033, 0.021),
            (0.115, -0.021, 0.023, 0.015),
            (0.135, -0.021, 0.008, 0.007),
        ],
    )
    ankle_soft_cuff_mesh = sleeve_z_mesh_xml(
        "ankle_soft_cuff_mesh",
        [
            (-0.014, 0.030, 0.028),
            (-0.008, 0.035, 0.033),
            (0.042, 0.037, 0.034),
            (0.052, 0.032, 0.030),
        ],
    )
    ankle_soft_strap_mesh = sleeve_z_mesh_xml(
        "ankle_soft_strap_mesh",
        [
            (0.022, 0.036, 0.033),
            (0.025, 0.039, 0.036),
            (0.040, 0.039, 0.036),
            (0.043, 0.036, 0.033),
        ],
    )
    ankle_lower_strap_mesh = sleeve_z_mesh_xml(
        "ankle_lower_strap_mesh",
        [
            (-0.010, 0.033, 0.031),
            (-0.007, 0.036, 0.034),
            (0.008, 0.036, 0.034),
            (0.011, 0.034, 0.032),
        ],
    )
    human_mesh_assets = textwrap.indent(
        "\n".join(
            [
                human_thigh_mesh,
                human_shank_mesh,
                human_foot_mesh,
                ankle_soft_cuff_mesh,
                ankle_soft_strap_mesh,
                ankle_lower_strap_mesh,
            ]
        ),
        "    ",
    )

    # ── 보행 보조 로봇 모델 ──────────────────────────────────────────────
    if robot_type == "exoskeleton":
        # 사람 관절과 같은 축/분절 길이를 갖는 측면 장착형 외골격.
        # 시각적 cuff는 외골격 링크에서 사람 분절 중심선까지 이어진다.
        # 고정 카메라가 -Y 쪽에 있으므로 외골격을 사람 앞쪽(-Y)에 배치한다.
        exo_y = -abs(exo_lateral_offset)
        exo_rod_radius = 0.012
        motor_radius = 0.0445
        motor_length = 0.05025
        motor_half_width = 0.5 * motor_length
        motor_inertia = cylinder_y_inertia_str(
            exo_motor_mass,
            motor_radius,
            motor_length,
        )
        exo_motor1_visual = actuator_visual_xml(
            "hip_motor", motor_radius, motor_half_width, "          "
        )
        exo_motor2_visual = actuator_visual_xml(
            "knee_motor", motor_radius, motor_half_width, "            "
        )
        exo_output_y = -motor_half_width - 0.010
        exo_thigh_visual = structural_link_visual_xml(
            "exo_thigh", thigh_length, exo_output_y, "        "
        )
        exo_shank_visual = structural_link_visual_xml(
            "exo_shank", shank_length, exo_output_y, "          "
        )
        thigh_cuff_z = -0.55 * thigh_length
        shank_cuff_z = -0.55 * shank_length
        ankle_connector_y = max(0.0, -exo_y - 0.037)

        arm_xml = f"""\
    <!-- ══════════════════════════════════════════════════════════
         Hip-Knee Exoskeleton
         motor1_joint: hip actuator, coaxial with human hip
         motor2_joint: knee actuator, coaxial with human knee
         group=1: exoskeleton visualization
    ═══════════════════════════════════════════════════════════ -->
    <body name="exoskeleton_base" pos="0 {exo_y:.4f} 0">
      <geom name="hip_mount_geom" type="capsule"
            fromto="0 0 0 0 {-exo_y:.4f} 0"
            size="0.012" rgba="0.25 0.25 0.25 1" group="1"
            contype="0" conaffinity="0"/>

      <body name="exo_thigh" pos="0 0 0">
        <joint name="motor1_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="0 0 {-0.5 * thigh_length:.4f}"
                  mass="{exo_link_mass:.4f}"
                  diaginertia="{inertia_str(exo_link_mass, thigh_length, exo_rod_radius)}"/>
        <body name="exo_motor1_mass" pos="0 0 0">
          <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                    diaginertia="{motor_inertia}"/>
{exo_motor1_visual}
        </body>
{exo_thigh_visual}
        <geom name="thigh_cuff_geom" type="capsule"
              fromto="0 0 {thigh_cuff_z:.4f} 0 {-exo_y:.4f} {thigh_cuff_z:.4f}"
              size="0.014" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>

        <body name="exo_shank" pos="0 0 {-thigh_length:.4f}">
          <joint name="motor2_joint" type="hinge" axis="0 1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * shank_length:.4f}"
                    mass="{exo_link_mass:.4f}"
                    diaginertia="{inertia_str(exo_link_mass, shank_length, exo_rod_radius)}"/>
          <body name="exo_motor2_mass" pos="0 0 0">
            <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                      diaginertia="{motor_inertia}"/>
{exo_motor2_visual}
          </body>
          <geom name="knee_cuff_geom" type="capsule"
                fromto="0 0 0 0 {-exo_y:.4f} 0"
                size="0.014" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
{exo_shank_visual}
          <geom name="shank_cuff_geom" type="capsule"
                fromto="0 0 {shank_cuff_z:.4f} 0 {-exo_y:.4f} {shank_cuff_z:.4f}"
                size="0.014" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
          <geom name="ankle_cuff_geom" type="capsule"
                fromto="0 0 {-shank_length:.4f} 0 {ankle_connector_y:.4f} {-shank_length:.4f}"
                size="0.011" material="mat_cuff_connector" group="1"
                contype="0" conaffinity="0"/>
          <site name="exo_ankle_site" pos="0 0 {-shank_length:.4f}"
                size="0.018" rgba="1.0 0.2 0.2 0"/>
          <site name="bracket_end_site" pos="0 {-exo_y:.4f} {-shank_length:.4f}"
                rgba="1 1 1 0"/>
        </body>
      </body>
    </body>
"""
    elif robot_type == "open_chain":
        # Exoskeleton과 같은 hip-side 장착 방식이지만 robot elbow는 사람
        # knee에 결합하지 않는다. 직렬 2-link 끝단 위치만 ankle에 연결한다.
        cuff_y = abs(exo_lateral_offset)
        link_radius = 0.012
        motor_radius = 0.0445
        motor_length = 0.05025
        motor_half_width = 0.5 * motor_length
        motor_inertia = cylinder_y_inertia_str(
            exo_motor_mass,
            motor_radius,
            motor_length,
        )
        open_chain_motor1_visual = actuator_visual_xml(
            "open_chain_motor1", motor_radius, motor_half_width, "          "
        )
        open_chain_motor2_visual = actuator_visual_xml(
            "open_chain_motor2", motor_radius, motor_half_width, "            "
        )
        open_chain_output_y = -motor_half_width - 0.010
        open_chain_link1_visual = structural_link_visual_xml(
            "open_chain_link1", link1_length, open_chain_output_y, "        "
        )
        open_chain_link2_visual = structural_link_visual_xml(
            "open_chain_link2", link2_length, open_chain_output_y, "          "
        )
        ankle_connector_y = max(0.0, cuff_y - 0.037)

        arm_xml = f"""\
    <!-- ══════════════════════════════════════════════════════════
         Open-Chain Ankle-Tracking Robot Arm
         motor1: fixed at human hip height
         motor2: serial elbow, not coupled to the human knee
         end-effector: position-only ankle cuff connection
    ═══════════════════════════════════════════════════════════ -->
    <body name="open_chain_base"
          pos="{base_pos[0]:.4f} {base_pos[1]:.4f} {base_pos[2]:.4f}">
      <geom name="open_chain_hip_mount" type="capsule"
            fromto="0 0 0 0 {cuff_y:.4f} 0"
            size="0.012" rgba="0.25 0.25 0.25 1" group="1"
            contype="0" conaffinity="0"/>

      <body name="open_chain_link1" pos="0 0 0">
        <joint name="motor1_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="0 0 {-0.5 * link1_length:.4f}"
                  mass="{exo_link_mass:.4f}"
                  diaginertia="{inertia_str(exo_link_mass, link1_length, link_radius)}"/>
        <body name="open_chain_motor1_mass" pos="0 0 0">
          <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                    diaginertia="{motor_inertia}"/>
{open_chain_motor1_visual}
        </body>
{open_chain_link1_visual}

        <body name="open_chain_link2" pos="0 0 {-link1_length:.4f}">
          <joint name="motor2_joint" type="hinge" axis="0 -1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * link2_length:.4f}"
                    mass="{exo_link_mass:.4f}"
                    diaginertia="{inertia_str(exo_link_mass, link2_length, link_radius)}"/>
          <body name="open_chain_motor2_mass" pos="0 0 0">
            <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                      diaginertia="{motor_inertia}"/>
{open_chain_motor2_visual}
          </body>
{open_chain_link2_visual}
          <geom name="open_chain_ankle_cuff" type="capsule"
                fromto="0 0 {-link2_length:.4f} 0 {ankle_connector_y:.4f} {-link2_length:.4f}"
                size="0.011" material="mat_cuff_connector" group="1"
                contype="0" conaffinity="0"/>
          <site name="open_chain_elbow_site" pos="0 0 0"
                size="0.014" rgba="0.2 0.2 0.2 1"/>
          <site name="open_chain_ee_site" pos="0 0 {-link2_length:.4f}"
                size="0.018" rgba="1.0 0.2 0.2 0"/>
          <site name="bracket_end_site"
                pos="0 {cuff_y:.4f} {-link2_length:.4f}"
                size="0.012" rgba="1.0 0.8 0.1 0"/>
        </body>
      </body>
    </body>
"""
    elif robot_type == "remote_exoskeleton":
        # 기존 remote_parallelogram의 A-B-C-D/E 전달 원리를 hip 높이로
        # 올린 구조. A-B와 D-C는 다리 링크에 직교하는 뒤쪽 crossbar다.
        exo_y = -abs(exo_lateral_offset)
        short_side = remote_exo_joint_spacing
        top_link = remote_exo_crank_length
        exo_plate_x = -0.5 * remote_exo_joint_spacing
        exo_plate_z = math.sqrt(
            max(
                0.0,
                remote_exo_upper_link**2
                - (0.5 * remote_exo_joint_spacing) ** 2,
            )
        )
        exo_rod_radius = 0.010
        motor_radius = 0.0445
        motor_length = 0.05025
        pin_half_y = 0.5 * motor_length
        motor_inertia = cylinder_y_inertia_str(
            exo_motor_mass,
            motor_radius,
            motor_length,
        )
        plate_inertia = inertia_str(
            remote_exo_plate_mass,
            max(remote_exo_upper_link, remote_exo_joint_spacing),
            0.010,
        )
        coupler_inertia = inertia_str(
            remote_exo_coupler_mass,
            thigh_length,
            0.007,
        )
        crank_inertia = inertia_str(
            remote_exo_crank_mass,
            top_link,
            0.009,
        )
        rod_inertia = inertia_str(
            remote_exo_rod_mass,
            top_link,
            0.008,
        )
        thigh_cuff_z = -0.55 * thigh_length
        shank_cuff_z = -0.55 * shank_length

        arm_xml = f"""\
    <!-- ══════════════════════════════════════════════════════════
         Remote-Actuated Hip-Knee Exoskeleton

         A: motor1 at the human hip, directly drives exo_thigh A-D
         D: passive exo_knee_joint, coaxial with the human knee
         A-B-C-D: equal 110 mm posterior crossbars maintain a parallelogram
         E: triangular upper plate pin driven by the remote motor2
         motor2: fixed behind/above the hip; crank/rod remotely drives knee
    ═══════════════════════════════════════════════════════════ -->
    <body name="remote_exoskeleton_base" pos="0 {exo_y:.4f} 0">
      <geom name="remote_exo_hip_mount" type="capsule"
            fromto="0 0 0 0 {-exo_y:.4f} 0"
            size="0.012" rgba="0.25 0.25 0.25 1" group="1"
            contype="0" conaffinity="0"/>
      <geom name="remote_exo_motor_frame" type="capsule"
            fromto="0 0 0 {remote_exo_motor_dx:.4f} 0 {remote_exo_motor_dz:.4f}"
            size="0.014" rgba="0.35 0.35 0.35 1" group="1"
            contype="0" conaffinity="0"/>

      <!-- A-D: hip motor and exoskeleton thigh. -->
      <body name="remote_exo_thigh" pos="0 0 0">
        <joint name="motor1_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="0 0 {-0.5 * thigh_length:.4f}"
                  mass="{exo_link_mass:.4f}"
                  diaginertia="{inertia_str(exo_link_mass, thigh_length, exo_rod_radius)}"/>
        <body name="remote_exo_motor1_mass" pos="0 0 0">
          <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                    diaginertia="{motor_inertia}"/>
          <geom name="remote_exo_hip_motor" type="cylinder"
                fromto="0 {-pin_half_y:.4f} 0 0 {pin_half_y:.4f} 0"
                size="{motor_radius:.4f}" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
        </body>
        <geom name="remote_exo_thigh_geom" type="capsule"
              fromto="0 0 0 0 0 {-thigh_length:.4f}"
              size="{exo_rod_radius:.4f}" material="mat_arm" group="1"
              contype="0" conaffinity="0"/>
        <geom name="remote_exo_thigh_cuff" type="capsule"
              fromto="0 0 {thigh_cuff_z:.4f} 0 {-exo_y:.4f} {thigh_cuff_z:.4f}"
              size="0.014" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>
        <site name="remote_exo_A_site" pos="0 0 0" size="0.010"
              rgba="1 1 0.2 1"/>

        <!-- D: knee flexion uses +Y, matching the human knee convention. -->
        <body name="remote_exo_shank" pos="0 0 {-thigh_length:.4f}">
          <joint name="exo_knee_joint" type="hinge" axis="0 1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * shank_length:.4f}"
                    mass="{exo_link_mass:.4f}"
                    diaginertia="{inertia_str(exo_link_mass, shank_length, exo_rod_radius)}"/>
          <geom name="remote_exo_knee_housing" type="cylinder"
                fromto="0 {-pin_half_y:.4f} 0 0 {pin_half_y:.4f} 0"
                size="0.016" rgba="0.08 0.08 0.08 1" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_exo_knee_cuff" type="capsule"
                fromto="0 0 0 0 {-exo_y:.4f} 0"
                size="0.014" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_exo_shank_geom" type="capsule"
                fromto="0 0 0 0 0 {-shank_length:.4f}"
                size="{exo_rod_radius:.4f}" material="mat_arm" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_exo_shank_cuff" type="capsule"
                fromto="0 0 {shank_cuff_z:.4f} 0 {-exo_y:.4f} {shank_cuff_z:.4f}"
                size="0.014" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_exo_ankle_cuff" type="capsule"
                fromto="0 0 {-shank_length:.4f} 0 {-exo_y:.4f} {-shank_length:.4f}"
                size="0.014" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_exo_DC_plate" type="capsule"
                fromto="0 0 0 {-short_side:.4f} 0 0"
                size="0.012" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
          <site name="remote_exo_D_site" pos="0 0 0" size="0.010"
                rgba="0.2 1.0 0.2 1"/>
          <site name="remote_exo_C_site" pos="{-short_side:.4f} 0 0"
                size="0.010" rgba="0.2 1.0 0.2 1"/>
          <site name="exo_ankle_site" pos="0 0 {-shank_length:.4f}"
                size="0.018" rgba="1.0 0.2 0.2 1.0"/>
          <site name="bracket_end_site"
                pos="0 {-exo_y:.4f} {-shank_length:.4f}"/>
        </body>
      </body>

      <!-- A-B-E plate: posterior crossbar and triangular drive tab. -->
      <body name="remote_exo_plate" pos="0 0 0">
        <joint name="plate_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="{0.5 * exo_plate_x:.4f} 0 {0.5 * exo_plate_z:.4f}"
                  mass="{remote_exo_plate_mass:.4f}"
                  diaginertia="{plate_inertia}"/>
        <geom name="remote_exo_AB_plate" type="capsule"
              fromto="0 0 0 {-short_side:.4f} 0 0"
              size="0.010" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>
        <geom name="remote_exo_AE_plate" type="capsule"
              fromto="0 0 0 {exo_plate_x:.4f} 0 {exo_plate_z:.4f}"
              size="0.010" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>
        <geom name="remote_exo_BE_plate" type="capsule"
              fromto="{-short_side:.4f} 0 0 {exo_plate_x:.4f} 0 {exo_plate_z:.4f}"
              size="0.010" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>
        <site name="remote_exo_E_site"
              pos="{exo_plate_x:.4f} 0 {exo_plate_z:.4f}"
              size="0.011" rgba="1.0 0.5 0.1 1.0"/>

        <!-- B-C long link remains parallel to the exoskeleton thigh. -->
        <body name="remote_exo_BC_link" pos="{-short_side:.4f} 0 0">
          <joint name="coupler_joint" type="hinge" axis="0 -1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * thigh_length:.4f}"
                    mass="{remote_exo_coupler_mass:.4f}"
                    diaginertia="{coupler_inertia}"/>
          <geom name="remote_exo_BC_geom" type="capsule"
                fromto="0 0 0 0 0 {-thigh_length:.4f}"
                size="0.007" material="mat_arm" group="1"
                contype="0" conaffinity="0"/>
          <site name="remote_exo_C_coupler_site"
                pos="0 0 {-thigh_length:.4f}"
                size="0.010" rgba="0.2 1.0 0.2 1"/>
        </body>
      </body>

      <!-- Motor2 is fixed behind/above the hip and drives E through crank/rod. -->
      <body name="remote_exo_crank"
            pos="{remote_exo_motor_dx:.4f} 0 {remote_exo_motor_dz:.4f}">
        <joint name="motor2_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="0 0 {-0.5 * top_link:.4f}"
                  mass="{remote_exo_crank_mass:.4f}"
                  diaginertia="{crank_inertia}"/>
        <body name="remote_exo_motor2_mass" pos="0 0 0">
          <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                    diaginertia="{motor_inertia}"/>
          <geom name="remote_exo_knee_motor" type="cylinder"
                fromto="0 {-pin_half_y:.4f} 0 0 {pin_half_y:.4f} 0"
                size="{motor_radius:.4f}" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
        </body>
        <geom name="remote_exo_crank_geom" type="capsule"
              fromto="0 0 0 0 0 {-top_link:.4f}"
              size="0.009" rgba="0.8 0.5 0.2 1" group="1"
              contype="0" conaffinity="0"/>

        <body name="remote_exo_rod" pos="0 0 {-top_link:.4f}">
          <joint name="rod_joint" type="hinge" axis="0 -1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * top_link:.4f}"
                    mass="{remote_exo_rod_mass:.4f}"
                    diaginertia="{rod_inertia}"/>
          <geom name="remote_exo_rod_geom" type="capsule"
                fromto="0 0 0 0 0 {-top_link:.4f}"
                size="0.008" rgba="0.8 0.3 0.2 1" group="1"
                contype="0" conaffinity="0"/>
          <site name="remote_exo_rod_end_site"
                pos="0 0 {-top_link:.4f}"
                size="0.011" rgba="1.0 0.5 0.1 1.0"/>
        </body>
      </body>
    </body>
"""
    elif robot_type == "remote_ankle_arm":
        # remote_exoskeleton의 posterior crossbar/crank 구조를 유지하되,
        # 사람 thigh/shank와의 관절 결합을 제거한 독립 2-link 발목 추종 팔.
        short_side = remote_exo_joint_spacing
        top_link = remote_exo_crank_length
        plate_x = -0.5 * short_side
        plate_z = math.sqrt(
            max(
                0.0,
                remote_exo_upper_link**2 - (0.5 * short_side) ** 2,
            )
        )
        cuff_y = abs(exo_lateral_offset)
        rod_radius = 0.010
        motor_radius = 0.0445
        motor_length = 0.05025
        pin_half_y = 0.5 * motor_length
        motor_inertia = cylinder_y_inertia_str(
            exo_motor_mass,
            motor_radius,
            motor_length,
        )
        plate_inertia = inertia_str(
            remote_exo_plate_mass,
            max(remote_exo_upper_link, short_side),
            0.010,
        )
        coupler_inertia = inertia_str(
            remote_exo_coupler_mass,
            link1_length,
            0.007,
        )
        crank_inertia = inertia_str(
            remote_exo_crank_mass,
            top_link,
            0.009,
        )
        rod_inertia = inertia_str(
            remote_exo_rod_mass,
            top_link,
            0.008,
        )

        arm_xml = f"""\
    <!-- ══════════════════════════════════════════════════════════
         Remote Ankle-Tracking Arm

         A-D/B-C: independent link1 parallelogram, not attached to thigh
         D-ankle: independent link2, not attached to shank
         A-B/D-C: equal posterior crossbars
         motor2: fixed behind/above A and remotely drives the distal angle
         ankle cuff: position-only connection; ankle rotation remains free
    ═══════════════════════════════════════════════════════════ -->
    <body name="remote_ankle_base"
          pos="{base_pos[0]:.4f} {base_pos[1]:.4f} {base_pos[2]:.4f}">
      <geom name="remote_ankle_hip_mount" type="capsule"
            fromto="0 0 0 0 {cuff_y:.4f} 0"
            size="0.012" rgba="0.25 0.25 0.25 1" group="1"
            contype="0" conaffinity="0"/>
      <geom name="remote_ankle_motor_frame" type="capsule"
            fromto="0 0 0 {remote_exo_motor_dx:.4f} 0 {remote_exo_motor_dz:.4f}"
            size="0.014" rgba="0.35 0.35 0.35 1" group="1"
            contype="0" conaffinity="0"/>

      <!-- A-D: independently sized motor1 drive link. -->
      <body name="remote_ankle_drive_link" pos="0 0 0">
        <joint name="motor1_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="0 0 {-0.5 * link1_length:.4f}"
                  mass="{exo_link_mass:.4f}"
                  diaginertia="{inertia_str(exo_link_mass, link1_length, rod_radius)}"/>
        <body name="remote_ankle_motor1_mass" pos="0 0 0">
          <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                    diaginertia="{motor_inertia}"/>
          <geom name="remote_ankle_motor1" type="cylinder"
                fromto="0 {-pin_half_y:.4f} 0 0 {pin_half_y:.4f} 0"
                size="{motor_radius:.4f}" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
        </body>
        <geom name="remote_ankle_AD_geom" type="capsule"
              fromto="0 0 0 0 0 {-link1_length:.4f}"
              size="{rod_radius:.4f}" material="mat_arm" group="1"
              contype="0" conaffinity="0"/>
        <site name="remote_ankle_A_site" pos="0 0 0" size="0.010"
              rgba="1 1 0.2 1"/>

        <!-- D: passive distal joint; link2 terminates at the ankle cuff. -->
        <body name="remote_ankle_distal_link"
              pos="0 0 {-link1_length:.4f}">
          <joint name="distal_joint" type="hinge" axis="0 -1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * link2_length:.4f}"
                    mass="{exo_link_mass:.4f}"
                    diaginertia="{inertia_str(exo_link_mass, link2_length, rod_radius)}"/>
          <geom name="remote_ankle_D_joint" type="cylinder"
                fromto="0 {-pin_half_y:.4f} 0 0 {pin_half_y:.4f} 0"
                size="0.016" rgba="0.08 0.08 0.08 1" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_ankle_link2_geom" type="capsule"
                fromto="0 0 0 0 0 {-link2_length:.4f}"
                size="{rod_radius:.4f}" material="mat_arm" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_ankle_DC_plate" type="capsule"
                fromto="0 0 0 {-short_side:.4f} 0 0"
                size="0.012" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
          <geom name="remote_ankle_cuff" type="capsule"
                fromto="0 0 {-link2_length:.4f} 0 {cuff_y:.4f} {-link2_length:.4f}"
                size="0.014" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
          <site name="remote_ankle_D_site" pos="0 0 0" size="0.010"
                rgba="0.2 1.0 0.2 1"/>
          <site name="remote_ankle_C_site" pos="{-short_side:.4f} 0 0"
                size="0.010" rgba="0.2 1.0 0.2 1"/>
          <site name="remote_ankle_ee_site"
                pos="0 0 {-link2_length:.4f}"
                size="0.018" rgba="1.0 0.2 0.2 1.0"/>
          <site name="bracket_end_site"
                pos="0 {cuff_y:.4f} {-link2_length:.4f}"
                size="0.012" rgba="1.0 0.8 0.1 1.0"/>
        </body>
      </body>

      <!-- A-B-E posterior plate and B-C parallel link. -->
      <body name="remote_ankle_plate" pos="0 0 0">
        <joint name="plate_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="{0.5 * plate_x:.4f} 0 {0.5 * plate_z:.4f}"
                  mass="{remote_exo_plate_mass:.4f}"
                  diaginertia="{plate_inertia}"/>
        <geom name="remote_ankle_AB_plate" type="capsule"
              fromto="0 0 0 {-short_side:.4f} 0 0"
              size="0.010" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>
        <geom name="remote_ankle_AE_plate" type="capsule"
              fromto="0 0 0 {plate_x:.4f} 0 {plate_z:.4f}"
              size="0.010" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>
        <geom name="remote_ankle_BE_plate" type="capsule"
              fromto="{-short_side:.4f} 0 0 {plate_x:.4f} 0 {plate_z:.4f}"
              size="0.010" rgba="0.15 0.35 0.85 1" group="1"
              contype="0" conaffinity="0"/>
        <site name="remote_ankle_E_site"
              pos="{plate_x:.4f} 0 {plate_z:.4f}"
              size="0.011" rgba="1.0 0.5 0.1 1.0"/>

        <body name="remote_ankle_BC_link"
              pos="{-short_side:.4f} 0 0">
          <joint name="coupler_joint" type="hinge" axis="0 -1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * link1_length:.4f}"
                    mass="{remote_exo_coupler_mass:.4f}"
                    diaginertia="{coupler_inertia}"/>
          <geom name="remote_ankle_BC_geom" type="capsule"
                fromto="0 0 0 0 0 {-link1_length:.4f}"
                size="0.007" material="mat_arm" group="1"
                contype="0" conaffinity="0"/>
          <site name="remote_ankle_C_coupler_site"
                pos="0 0 {-link1_length:.4f}"
                size="0.010" rgba="0.2 1.0 0.2 1"/>
        </body>
      </body>

      <!-- Rear motor2 drives E through equal-length crank and rod. -->
      <body name="remote_ankle_crank"
            pos="{remote_exo_motor_dx:.4f} 0 {remote_exo_motor_dz:.4f}">
        <joint name="motor2_joint" type="hinge" axis="0 -1 0"
               pos="0 0 0" damping="0.1"/>
        <inertial pos="0 0 {-0.5 * top_link:.4f}"
                  mass="{remote_exo_crank_mass:.4f}"
                  diaginertia="{crank_inertia}"/>
        <body name="remote_ankle_motor2_mass" pos="0 0 0">
          <inertial pos="0 0 0" mass="{exo_motor_mass:.4f}"
                    diaginertia="{motor_inertia}"/>
          <geom name="remote_ankle_motor2" type="cylinder"
                fromto="0 {-pin_half_y:.4f} 0 0 {pin_half_y:.4f} 0"
                size="{motor_radius:.4f}" rgba="0.15 0.35 0.85 1" group="1"
                contype="0" conaffinity="0"/>
        </body>
        <geom name="remote_ankle_crank_geom" type="capsule"
              fromto="0 0 0 0 0 {-top_link:.4f}"
              size="0.009" rgba="0.8 0.5 0.2 1" group="1"
              contype="0" conaffinity="0"/>

        <body name="remote_ankle_rod" pos="0 0 {-top_link:.4f}">
          <joint name="rod_joint" type="hinge" axis="0 -1 0"
                 pos="0 0 0" damping="0.1"/>
          <inertial pos="0 0 {-0.5 * top_link:.4f}"
                    mass="{remote_exo_rod_mass:.4f}"
                    diaginertia="{rod_inertia}"/>
          <geom name="remote_ankle_rod_geom" type="capsule"
                fromto="0 0 0 0 0 {-top_link:.4f}"
                size="0.008" rgba="0.8 0.3 0.2 1" group="1"
                contype="0" conaffinity="0"/>
          <site name="remote_ankle_rod_end_site"
                pos="0 0 {-top_link:.4f}"
                size="0.011" rgba="1.0 0.5 0.1 1.0"/>
        </body>
      </body>
    </body>
"""
    elif robot_type == "remote_parallelogram":
        # ──────────────────────────────────────────────────────────────
        # Remote parallelogram - A/B/C/D/E topology revised from user sketch
        #
        # User-defined points:
        #   A : motor1 axis. Two coaxial rotations exist at A.
        #       - motor1_joint(active) drives link A-D.
        #       - plate_joint(passive) carries the rigid triangular plate A-B-E.
        #   B : passive joint on the A-B-E plate. Link B-C is attached here.
        #   C : distal passive joint at the end of a fixed standoff mounted on the EE link.
        #   D : distal passive joint at the end of another fixed standoff mounted on the EE link.
        #   E : passive pin on the A-B-E plate connected to the motor2 crank-rocker linkage.
        #
        # Parallelogram:
        #   A-B-C-D forms the parallelogram.
        #   A-D and B-C are the two long parallel links.
        #   A-B and D-C are the two short parallel links.
        #
        # Distal end:
        #   C and D are NOT directly on the EE link. The EE link has two fixed capture links;
        #   C and D passive pins are mounted at the capture-link tips.
        #   A-D is the motor1-driven link and is placed on the end-effector side.
        # ──────────────────────────────────────────────────────────────
        mz = motor_offset_z
        md = motor_distance
        short_side = remote_e_motor1_joint_offset
        top_link = remote_c_top_link
        standoff = distal_standoff        # fixed capture-link length from ee_foot_geom to C/D passive pins

        # E point on triangular plate A-B-E.
        # A-B=e, A-E=d, B-E=d. E is therefore computed from the d,d,e triangle.
        Ex = plate_e_local_x
        Ez = plate_e_local_z

        coupler_len = link1_length         # |B-C| = |A-D|
        r_rod = 0.007
        r_blue = 0.010
        r_pin = 0.006
        capture_overlap = 2.0 * r_rod

        # Visual layer offsets only. Joint axes remain coaxial/parallel in the X-Z plane.
        layer = abs(rod_layer_offset)
        layer = 0.0
        drive_rod_y = -layer               # A-D motor1 rod: inner side
        coupler_rod_y = layer              # B-C passive rod: outer side
        mid_y = 0.5 * (drive_rod_y + coupler_rod_y)
        pin_half_y = layer + 0.014
        # pin_half_y = 0

        # Distal EE link local frame:
        #   origin = D passive joint. D is the distal pin connected to the motor1-driven A-D link.
        #   local +Z = link2 direction.
        #   local +X = perpendicular standoff direction from the C/D passive pins to link2.
        # C and D are at the tips of two short perpendicular standoffs attached to link2.
        ee_root_x = standoff
        ee_root_z = -short_side - capture_overlap
        Fx = standoff
        Fz = link2_length

        arm_xml = f"""\
    <!-- ══════════════════════════════════════════════════════════
         Robot Arm - Remote Parallelogram, A-B-C-D with triangular plate A-B-E

         A: motor1 active joint + coaxial passive plate_joint
         B: passive joint on triangular plate A-B-E
         C: distal passive joint on fixed EE standoff
         D: distal passive joint on fixed EE standoff
         E: passive pin connected to motor2 crank-rocker

         Distal closure:
           link2 is one rigid link.
           C/D passive pins are at the ends of two short perpendicular standoffs attached to link2.
    ═══════════════════════════════════════════════════════════ -->
    <body name="arm_base" pos="{base_pos[0]:.4f} {base_pos[1]:.4f} {base_pos[2]:.4f}">
      <!-- <geom name="passive_pin" type="cylinder" fromto="-0.04 0 0 0.04 0 0" size="0.012" rgba="0.8 0.2 0.2 1" group="1" contype="0" conaffinity="0"/> -->
      <body name="passive_bracket" pos="0 0 0">
        <joint name="base_passive_joint" type="hinge" axis="1 0 0" pos="0 0 0" damping="{passive_damping}"/>
        <inertial pos="0 0 {-mz/2:.4f}" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>

        <!-- A-D: motor1 directly drives this long rod. D is the first distal passive joint. -->
        <body name="AD_drive_link" pos="0 0 {-mz:.4f}">
          <joint name="motor1_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.1"/>
          <geom name="motor1_geom" type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.020" rgba="0.15 0.35 0.85 1" group="1" contype="0" conaffinity="0"/>
          <inertial pos="0 0 {link1_com_z:.4f}" mass="{link1_mass:.4f}" diaginertia="{link1_inertia}"/>
          <geom name="AD_drive_rod_geom" type="capsule" fromto="0 {drive_rod_y:.4f} 0 0 {drive_rod_y:.4f} {-link1_length:.4f}" size="{r_rod}" material="mat_arm" group="1"/>
          <site name="A_site" pos="0 0 0" size="0.010" rgba="1 1 0.2 1"/>
          <site name="D_drive_tip_site" pos="0 0 {-link1_length:.4f}" size="0.010" rgba="0.2 1.0 0.2 1.0"/>

          <!-- D passive joint at the tip of the D capture link.
               The capture links below are NOT joints/bodies. They are fixed geoms of the EE link,
               so they cannot rotate independently. Only the passive pin axes at C/D rotate. -->
          <body name="distal_ee_link" pos="0 0 {-link1_length:.4f}">
            <joint name="distal_D_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.1"/>
            <inertial pos="{0.35*Fx:.4f} {mid_y:.4f} {0.35*Fz:.4f}" mass="{link2_mass:.4f}" diaginertia="{link2_inertia}"/>

            <!-- End-effector link2. C and D are not directly on it; each is reached by a
                 short perpendicular standoff fixed to this same distal_ee_link body. -->
            <geom name="ee_foot_geom" type="capsule"
                  fromto="{ee_root_x:.4f} {mid_y:.4f} {ee_root_z:.4f} {Fx:.4f} {mid_y:.4f} {Fz:.4f}"
                  size="{r_rod}" material="mat_arm" group="1"/>

            <!-- Fixed perpendicular standoff #1: link2 -> D passive pin. -->
            <geom name="D_capture_link_geom" type="capsule"
                  fromto="{standoff:.4f} {drive_rod_y:.4f} 0 0 {drive_rod_y:.4f} 0"
                  size="{r_blue}" rgba="0.15 0.35 0.85 1" group="1"/>
            <geom name="D_passive_pin_geom" type="cylinder"
                  fromto="0 {-pin_half_y:.4f} 0 0 {pin_half_y:.4f} 0"
                  size="{r_pin}" rgba="0.05 0.05 0.05 1" group="1" contype="0" conaffinity="0"/>

            <!-- Fixed perpendicular standoff #2: link2 -> C passive pin. -->
            <geom name="C_capture_link_geom" type="capsule"
                  fromto="{standoff:.4f} {coupler_rod_y:.4f} {-short_side:.4f} 0 {coupler_rod_y:.4f} {-short_side:.4f}"
                  size="{r_blue}" rgba="0.15 0.35 0.85 1" group="1"/>
            <geom name="C_passive_pin_geom" type="cylinder"
                  fromto="0 {-pin_half_y:.4f} {-short_side:.4f} 0 {pin_half_y:.4f} {-short_side:.4f}"
                  size="{r_pin}" rgba="0.05 0.05 0.05 1" group="1" contype="0" conaffinity="0"/>

            <site name="D_joint_site" pos="0 0 0" size="0.010" rgba="0.2 1.0 0.2 1.0"/>
            <site name="ee_C_site" pos="0 0 {-short_side:.4f}" size="0.010" rgba="0.2 1.0 0.2 1.0"/>
            <site name="ee_site" pos="{Fx:.4f} {mid_y:.4f} {Fz:.4f}" size="0.020" rgba="1.0 0.2 0.2 1.0"/>

            <geom name="bracket_geom" type="cylinder"
                  fromto="{Fx:.4f} {mid_y:.4f} {Fz:.4f} {Fx:.4f} {mid_y - cuff_length:.4f} {Fz:.4f}"
                  size="0.010" material="mat_arm" group="1"/>
            <site name="bracket_end_site" pos="{Fx:.4f} {mid_y - cuff_length:.4f} {Fz:.4f}"/>
          </body>
        </body>

        <!-- A-B-E triangular plate: passive and coaxial with motor1 at A. -->
        <body name="ABE_passive_plate" pos="0 0 {-mz:.4f}">
          <joint name="plate_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.1"/>
          <inertial pos="{0.5*Ex:.4f} 0 {-0.5*short_side:.4f}" mass="{aux_mass:.4f}" diaginertia="{aux_inertia}"/>

          <geom name="plate_AB_geom" type="capsule" fromto="0 {mid_y:.4f} 0 0 {mid_y:.4f} {-short_side:.4f}" size="{r_blue}" rgba="0.15 0.35 0.85 1" group="1"/>
          <geom name="plate_AE_geom" type="capsule" fromto="0 0 0 {Ex:.4f} 0 {Ez:.4f}" size="{r_blue}" rgba="0.15 0.35 0.85 1" group="1"/>
          <geom name="plate_BE_geom" type="capsule" fromto="0 {coupler_rod_y:.4f} {-short_side:.4f} {Ex:.4f} 0 {Ez:.4f}" size="{r_blue}" rgba="0.15 0.35 0.85 1" group="1"/>

          <site name="plate_A_site" pos="0 0 0" size="0.010" rgba="1 1 0.2 1"/>
          <site name="plate_B_site" pos="0 0 {-short_side:.4f}" size="0.010" rgba="0.2 1.0 0.2 1.0"/>
          <site name="plate_E_site" pos="{Ex:.4f} 0 {Ez:.4f}" size="0.012" rgba="1.0 0.5 0.1 1.0"/>
          <geom name="plate_E_pin_geom" type="cylinder"
                fromto="{Ex:.4f} {-pin_half_y:.4f} {Ez:.4f} {Ex:.4f} {pin_half_y:.4f} {Ez:.4f}"
                size="{r_pin}" rgba="0.05 0.05 0.05 1" group="1" contype="0" conaffinity="0"/>

          <!-- B-C: passive coupler link. C is a passive pin connected to the EE-side C bracket. -->
          <body name="BC_coupler_link" pos="0 0 {-short_side:.4f}">
            <joint name="coupler_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.1"/>
            <inertial pos="0 0 {-coupler_len/2:.4f}" mass="{aux_mass:.4f}" diaginertia="{aux_inertia}"/>
            <geom name="BC_coupler_rod_geom" type="capsule" fromto="0 {coupler_rod_y:.4f} 0 0 {coupler_rod_y:.4f} {-coupler_len:.4f}" size="{r_rod}" material="mat_arm" group="1"/>
            <site name="C_coupler_tip_site" pos="0 0 {-coupler_len:.4f}" size="0.010" rgba="0.2 1.0 0.2 1.0"/>

            <body name="distal_C_pin" pos="0 0 {-coupler_len:.4f}">
              <joint name="distal_C_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.1"/>
              <inertial pos="0 0 0" mass="{aux_mass*0.2:.4f}" diaginertia="{aux_inertia}"/>
              <!-- Visual C pin is fixed at the end of C_capture_link_geom on the EE body. This body only supplies the passive hinge DOF. -->
              <site name="distal_C_joint_site" pos="0 0 0" size="0.010" rgba="0.2 1.0 0.2 1.0"/>
            </body>
          </body>
        </body>

        <!-- motor2 crank-rocker: connects to E on the A-B-E triangular plate. -->
        <body name="crank" pos="{md:.4f} 0 {-mz + motor2_offset_z:.4f}">
          <joint name="motor2_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.1"/>
          <geom name="motor2_geom" type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.020" rgba="0.2 0.2 0.8 1" group="1" contype="0" conaffinity="0"/>
          <inertial pos="0 0 {-top_link/2:.4f}" mass="{aux_mass:.4f}" diaginertia="{aux_inertia}"/>
          <geom name="crank_geom" type="capsule" fromto="0 0 0 0 0 {-top_link:.4f}" size="0.010" rgba="0.8 0.5 0.2 1" group="1"/>

          <body name="rod" pos="0 0 {-top_link:.4f}">
            <joint name="rod_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.1"/>
            <inertial pos="0 0 {-top_link/2:.4f}" mass="{aux_mass:.4f}" diaginertia="{aux_inertia}"/>
            <geom name="motor2_rod_geom" type="capsule" fromto="0 0 0 0 0 {-top_link:.4f}" size="0.009" rgba="0.8 0.3 0.2 1" group="1"/>
            <site name="rod_end_site" pos="0 0 {-top_link:.4f}" size="0.012" rgba="1.0 0.5 0.1 1.0"/>
          </body>
        </body>
      </body>
    </body>
"""

    passive_actuator_xml = ""
    passive_sensor_xml = ""
    if robot_type == "remote_parallelogram":
        passive_actuator_xml = (
            '    <position name="passive_act" joint="base_passive_joint" '
            'kp="5000" kv="100"/>\n'
        )
    if robot_type in {
        "remote_ankle_arm",
        "remote_exoskeleton",
        "remote_parallelogram",
    }:
        passive_sensor_xml = (
            '    <jointpos name="s_passive" joint="'
            + (
                "plate_joint"
                if robot_type in {"remote_ankle_arm", "remote_exoskeleton"}
                else "base_passive_joint"
            )
            + '"/>\n'
        )

    xml = f"""\
<?xml version="1.0" encoding="utf-8"?>
<!--
  leg_and_arm.xml
  사람 2-link 하지 + 보행 보조 로봇 통합 MuJoCo 모델.
  MuJoCo 3.3.0 호환.
  Auto-generated by build_xml.py — 직접 수정하지 말 것.
-->
<mujoco model="leg_and_arm">

  <compiler angle="radian" autolimits="true"/>

  <option timestep="{sim_timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>

  <!-- ── 시각 설정 ─────────────────────────────────────────────── -->
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <!-- ── 기본 재질/지오메트리 ──────────────────────────────────── -->
  <asset>
{human_mesh_assets}
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>
    <texture name="texplane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512" mark="cross" markrgb="0.8 0.8 0.8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="2 2" texuniform="true"/>
    
    <material name="mat_leg"   rgba="0.78 0.59 0.43 1.0" specular="0.18" shininess="0.22" reflectance="0.05"/>
    <material name="mat_leg_highlight" rgba="0.88 0.69 0.52 1.0" specular="0.22" shininess="0.25" reflectance="0.05"/>
    <material name="mat_shoe" rgba="0.10 0.13 0.17 1.0" specular="0.3" shininess="0.3"/>
    <material name="mat_arm"   rgba="0.2 0.2 0.2 1.0" reflectance="0.5"/>
    <material name="mat_link_structure" rgba="0.075 0.09 0.105 1" specular="0.7" shininess="0.6" reflectance="0.22"/>
    <material name="mat_link_accent" rgba="0.04 0.22 0.48 1" specular="0.55" shininess="0.5" reflectance="0.12"/>
    <material name="mat_link_pin" rgba="0.48 0.52 0.56 1" specular="0.9" shininess="0.75" reflectance="0.30"/>
    <material name="mat_joint" rgba="0.8 0.2 0.2 1.0" reflectance="0.3"/>
    <material name="mat_ee"    rgba="1.0 0.8 0.1 1.0"/>
    <!-- actuator 외형 전용 재질: 질량/충돌에는 영향을 주지 않는다. -->
    <material name="mat_motor_housing" rgba="0.055 0.075 0.095 1" specular="0.65" shininess="0.55" reflectance="0.25"/>
    <material name="mat_motor_cap" rgba="0.025 0.030 0.038 1" specular="0.35" shininess="0.35"/>
    <material name="mat_motor_band" rgba="0.05 0.20 0.43 1" specular="0.55" shininess="0.45" reflectance="0.15"/>
    <material name="mat_motor_metal" rgba="0.48 0.52 0.56 1" specular="0.9" shininess="0.75" reflectance="0.35"/>
    <material name="mat_motor_hub" rgba="0.12 0.14 0.16 1" specular="0.75" shininess="0.65"/>
    <material name="mat_motor_bolt" rgba="0.68 0.72 0.76 1" specular="0.95" shininess="0.85"/>
    <material name="mat_motor_connector" rgba="0.03 0.035 0.04 1" specular="0.2" shininess="0.2"/>
    <!-- 발목 soft cuff: 패딩 원단, 외측 체결 패드, 봉제선 -->
    <material name="mat_cuff_fabric" rgba="0.065 0.23 0.35 1" specular="0.07" shininess="0.08" reflectance="0.02"/>
    <material name="mat_cuff_trim" rgba="0.055 0.43 0.64 1" specular="0.16" shininess="0.15" reflectance="0.04"/>
    <material name="mat_cuff_stitch" rgba="0.58 0.76 0.86 1" specular="0.10" shininess="0.10"/>
    <material name="mat_cuff_connector" rgba="0.055 0.075 0.090 1" specular="0.30" shininess="0.28" reflectance="0.08"/>
    <!-- 고정식 소아 수동보행기 재질 -->
    <material name="mat_walker_blue" rgba="0.025 0.075 0.56 1" specular="0.68" shininess="0.58" reflectance="0.18"/>
    <material name="mat_walker_metal" rgba="0.40 0.44 0.48 1" specular="0.92" shininess="0.82" reflectance="0.30"/>
    <material name="mat_walker_clamp" rgba="0.075 0.085 0.095 1" specular="0.42" shininess="0.38" reflectance="0.08"/>
    <material name="mat_walker_grip" rgba="0.035 0.040 0.045 1" specular="0.15" shininess="0.16"/>
    <material name="mat_walker_pad" rgba="0.085 0.095 0.105 1" specular="0.18" shininess="0.20"/>
    <material name="mat_walker_rubber" rgba="0.025 0.028 0.032 1" specular="0.12" shininess="0.14"/>
    <material name="mat_walker_hub" rgba="0.58 0.62 0.66 1" specular="0.95" shininess="0.86" reflectance="0.34"/>
  </asset>

  <worldbody>
    <!-- 바닥 -->
    <geom name="floor" type="plane" pos="0 0 {floor_z:.3f}" size="3 3 0.1" material="matplane" group="0"/>

    <!-- ── 조명 ──────────────────────────────────────────────── -->
    <light name="top_light" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
    <light name="front_light" pos="2 -2 1.5" dir="-1 1 -0.5" diffuse="0.5 0.5 0.5"/>
    <light name="side_light" pos="0 3 1.5" dir="0 -1 -0.5" diffuse="0.4 0.4 0.4"/>

    <!-- ── 카메라 (비디오 저장용 측면 뷰) ───────────────────────── -->
    <camera name="fixed" pos="0.0 -1.45 -0.16" xyaxes="1 0 0 0 0 1"/>
    <!-- -Y 시상면에서 X 방향으로 약 24도 이동한 actuator 확인용 사선 시점. -->
    <camera name="presentation" pos="0.72 -1.55 -0.14"
            xyaxes="0.9119 0.4104 0 0 0 1"/>

    <!-- ── 참조선 (Global 원점 표시) ─────────────────────────── -->
    <site name="origin" pos="0 0 0" size="0.02" rgba="1 1 1 0.5"/>

    <!-- ══════════════════════════════════════════════════════════
         사람 하지 (Leg Model)
         group=0 : 시각화 그룹
    ═══════════════════════════════════════════════════════════ -->
    <body name="hip" pos="0 0 0">
      <!-- Hip 관절: Y축 회전, 양수=flexion -->
      <joint name="hip_joint" type="hinge" axis="0 -1 0" pos="0 0 0" damping="0.5"/>
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>

      <!-- Thigh -->
      <body name="thigh" pos="0 0 0">
        <inertial pos="0 0 {thigh_com_z:.4f}" mass="{thigh_mass:.4f}" diaginertia="{thigh_inertia}"/>
        <!-- 기존 capsule은 동일한 충돌 외형을 유지하되 투명하게 둔다. -->
        <geom name="thigh_geom" type="capsule" fromto="0 0 0 0 0 {-thigh_length:.4f}"
              size="{r_leg}" rgba="0 0 0 0" group="3"/>
        <!-- 연속 단면 mesh로 대퇴의 taper를 표현한다. -->
        <geom name="human_thigh_visual" type="mesh" mesh="human_thigh_mesh"
              material="mat_leg_highlight" group="0" contype="0" conaffinity="0"/>
        <geom name="human_patella_visual" type="ellipsoid"
              pos="0.020 0 {-thigh_length + 0.002:.4f}"
              size="0.018 0.027 0.022" material="mat_leg" group="0"
              contype="0" conaffinity="0"/>
        <geom name="human_knee_bridge" type="ellipsoid"
              pos="0.003 0 {-thigh_length:.4f}"
              size="0.034 0.031 0.022" material="mat_leg_highlight" group="0"
              contype="0" conaffinity="0"/>

        <!-- Knee 관절: Y축 회전, 양수=flexion (하퇴 후방 굽힘) -->
        <body name="shank" pos="0 0 {-thigh_length:.4f}">
          <joint name="knee_joint" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.5"/>
          <inertial pos="0 0 {shank_com_z:.4f}" mass="{shank_mass:.4f}" diaginertia="{shank_inertia}"/>
          <geom name="shank_geom" type="capsule" fromto="0 0 0 0 0 {-shank_length:.4f}"
                size="{r_leg}" rgba="0 0 0 0" group="3"/>
          <geom name="human_shank_visual" type="mesh" mesh="human_shank_mesh"
                material="mat_leg_highlight" group="0" contype="0" conaffinity="0"/>

          <!-- 사람 발목을 감싸는 부드러운 패딩 커프. 로봇 링크는 외측
               체결 패드까지만 닿고, 실제 위치 구속 site는 커프 중심에 둔다. -->
          <geom name="ankle_soft_cuff_visual" type="mesh" mesh="ankle_soft_cuff_mesh"
                pos="0 0 {-shank_length:.4f}" material="mat_cuff_fabric" group="1"
                contype="0" conaffinity="0"/>
          <geom name="ankle_soft_cuff_velcro_strap" type="mesh" mesh="ankle_soft_strap_mesh"
                pos="0 0 {-shank_length:.4f}" material="mat_cuff_trim" group="1"
                contype="0" conaffinity="0"/>
          <geom name="ankle_soft_cuff_lower_strap" type="mesh" mesh="ankle_lower_strap_mesh"
                pos="0 0 {-shank_length:.4f}" material="mat_cuff_trim" group="1"
                contype="0" conaffinity="0"/>
          <geom name="ankle_soft_cuff_side_pad" type="box"
                pos="0 -0.036 {-shank_length + 0.019:.4f}" size="0.021 0.006 0.027"
                material="mat_cuff_trim" group="1" contype="0" conaffinity="0"/>
          <geom name="ankle_soft_cuff_front_strap" type="capsule"
                fromto="0.036 -0.025 {-shank_length + 0.032:.4f} 0.036 0.025 {-shank_length + 0.032:.4f}"
                size="0.004" material="mat_cuff_trim" group="1"
                contype="0" conaffinity="0"/>
          <geom name="ankle_soft_cuff_upper_stitch" type="capsule"
                fromto="-0.022 -0.033 {-shank_length + 0.047:.4f} 0.022 -0.033 {-shank_length + 0.047:.4f}"
                size="0.0022" material="mat_cuff_stitch" group="1"
                contype="0" conaffinity="0"/>
          <geom name="ankle_soft_cuff_lower_stitch" type="capsule"
                fromto="-0.020 -0.032 {-shank_length - 0.008:.4f} 0.020 -0.032 {-shank_length - 0.008:.4f}"
                size="0.0022" material="mat_cuff_stitch" group="1"
                contype="0" conaffinity="0"/>

          <!-- Ankle site (end-effector 참조점) -->
          <site name="ankle_site" pos="0 0 {-shank_length:.4f}" size="0.005" rgba="0.2 1.0 0.2 0"/>
          
          <!-- Ball joint를 가진 더미 바디 (회전 충돌 방지용) -->
          <body name="cuff_dummy" pos="0 0 {-shank_length:.4f}">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
              <joint name="cuff_spherical" type="ball" damping="0.01"/>
              <site name="cuff_dummy_site" pos="0 0 0"/>
          </body>

          <!-- 발 질량 (ankle 하중) -->
          <body name="foot" pos="0 0 {ankle_rel_z:.4f}">
            <inertial pos="0 0 0" mass="{foot_mass:.4f}" diaginertia="1e-4 1e-4 1e-4"/>
            <geom name="human_foot_visual" type="mesh" mesh="human_foot_mesh"
                  material="mat_shoe" group="0"
                  contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>

    {arm_xml}

{walker_xml}

  </worldbody>

  <!-- ── Actuator 정의 ──────────────────────────────────────────────────── -->
  <actuator>
    <!-- 사람 다리: position actuator -->
    <!-- robot_drives_human 모드에서는 로봇이 사람을 끌고 가야하므로 사람의 근력(kp)을 0으로 만듦 -->
    <position name="hip_act"  joint="hip_joint"  kp="{0 if control_mode == 'robot_drives_human' else 5000}" kv="{1 if control_mode == 'robot_drives_human' else 100}"/>
    <position name="knee_act" joint="knee_joint" kp="{0 if control_mode == 'robot_drives_human' else 5000}" kv="{1 if control_mode == 'robot_drives_human' else 100}"/>

{passive_actuator_xml}
    <!-- 보조 로봇: general torque actuator (외부 PD 제어) -->
    <general name="motor1_act" joint="motor1_joint" gear="1" dyntype="none"/>
    <general name="motor2_act" joint="motor2_joint" gear="1" dyntype="none"/>
  </actuator>

  <!-- ── Sensor ─────────────────────────────────────────────────────────── -->
  <sensor>
    <jointpos  name="s_hip_pos"    joint="hip_joint"/>
    <jointpos  name="s_knee_pos"   joint="knee_joint"/>
{passive_sensor_xml}
    <jointpos  name="s_motor1"     joint="motor1_joint"/>
    <jointpos  name="s_motor2"     joint="motor2_joint"/>
    <jointvel  name="s_hip_vel"    joint="hip_joint"/>
    <jointvel  name="s_knee_vel"   joint="knee_joint"/>
    <jointvel  name="s_motor1_vel" joint="motor1_joint"/>
    <jointvel  name="s_motor2_vel" joint="motor2_joint"/>
    <framepos  name="s_ankle_pos"  objtype="site" objname="ankle_site"/>
    <framepos  name="s_ee_pos"     objtype="site" objname="bracket_end_site"/>
  </sensor>

"""
    # ── Tendon 정의: remote_parallelogram의 open-parallelogram 각도 폐쇄 ─────────
    if robot_type == "remote_parallelogram":
        xml += "\n  <tendon>\n"
        # abs(coupler_link) = abs(drive_link)
        #   abs(coupler) = plate_joint + coupler_joint
        #   abs(drive)   = motor1_joint
        # → plate_joint + coupler_joint - motor1_joint = 0
        xml += '    <fixed name="parallel_long_links_tendon">\n'
        xml += '      <joint joint="plate_joint" coef="1"/>\n'
        xml += '      <joint joint="coupler_joint" coef="1"/>\n'
        xml += '      <joint joint="motor1_joint" coef="-1"/>\n'
        xml += '    </fixed>\n'

        # abs(distal_ee_link) = abs(coaxial_passive_plate)
        #   abs(distal) = motor1_joint + distal_D_joint
        #   abs(plate)  = plate_joint
        # → motor1_joint + distal_D_joint - plate_joint = 0
        xml += '    <fixed name="parallel_short_links_tendon">\n'
        xml += '      <joint joint="motor1_joint" coef="1"/>\n'
        xml += '      <joint joint="distal_D_joint" coef="1"/>\n'
        xml += '      <joint joint="plate_joint" coef="-1"/>\n'
        xml += '    </fixed>\n'
        xml += "  </tendon>\n"
    elif robot_type in {"remote_ankle_arm", "remote_exoskeleton"}:
        xml += "\n  <tendon>\n"
        prefix = (
            "remote_ankle"
            if robot_type == "remote_ankle_arm"
            else "remote_exo"
        )
        # B-C는 A-D와 평행: plate + coupler = motor1.
        xml += f'    <fixed name="{prefix}_long_links_tendon">\n'
        xml += '      <joint joint="plate_joint" coef="1"/>\n'
        xml += '      <joint joint="coupler_joint" coef="1"/>\n'
        xml += '      <joint joint="motor1_joint" coef="-1"/>\n'
        xml += '    </fixed>\n'
        # distal link의 절대각은 plate 절대각과 같다.
        distal_joint_name = (
            "distal_joint"
            if robot_type == "remote_ankle_arm"
            else "exo_knee_joint"
        )
        distal_coef = "1" if robot_type == "remote_ankle_arm" else "-1"
        xml += f'    <fixed name="{prefix}_short_links_tendon">\n'
        xml += '      <joint joint="motor1_joint" coef="1"/>\n'
        xml += (
            f'      <joint joint="{distal_joint_name}" '
            f'coef="{distal_coef}"/>\n'
        )
        xml += '      <joint joint="plate_joint" coef="-1"/>\n'
        xml += '    </fixed>\n'
        xml += "  </tendon>\n"

    # ── 구속 조건 ────────────────────────────────────────────────────────
    xml += "\n  <equality>\n"
    if robot_type == "exoskeleton":
        # 외골격 관절과 사람 관절을 단단한 1:1 cuff 결합으로 묶는다.
        # 기본 equality solver(20 ms)는 빠른 knee motion에서 수 도의 상대
        # 변위를 허용하므로, 발목 connect와 같은 1 ms 설정을 사용한다.
        _sr = 'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"'
        xml += f'    <joint name="exo_hip_coupling" joint1="motor1_joint" joint2="hip_joint" polycoef="0 1 0 0 0" {_sr}/>\n'
        xml += f'    <joint name="exo_knee_coupling" joint1="motor2_joint" joint2="knee_joint" polycoef="0 1 0 0 0" {_sr}/>\n'
    elif robot_type == "open_chain":
        # 사람 knee와의 joint equality는 없고, 발목 위치만 연결한다.
        _sr = 'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"'
        xml += f'    <connect name="open_chain_ankle_constraint" site1="cuff_dummy_site" site2="bracket_end_site" {_sr}/>\n'
    elif robot_type == "remote_exoskeleton":
        # 사람 관절과 직접 결합되는 전달기구이므로 기존 EE형보다 단단한 폐루프를 사용한다.
        _sr = 'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"'
        xml += f'    <connect name="remote_exo_loop_crank" site1="remote_exo_rod_end_site" site2="remote_exo_E_site" {_sr}/>\n'
        xml += f'    <connect name="remote_exo_loop_C" site1="remote_exo_C_coupler_site" site2="remote_exo_C_site" {_sr}/>\n'
        xml += f'    <tendon name="remote_exo_parallel_long" tendon1="remote_exo_long_links_tendon" polycoef="0 0 0 0 0" {_sr}/>\n'
        xml += f'    <tendon name="remote_exo_parallel_short" tendon1="remote_exo_short_links_tendon" polycoef="0 0 0 0 0" {_sr}/>\n'
        xml += '    <joint name="remote_exo_hip_coupling" joint1="motor1_joint" joint2="hip_joint" polycoef="0 1 0 0 0"/>\n'
        xml += '    <joint name="remote_exo_knee_coupling" joint1="exo_knee_joint" joint2="knee_joint" polycoef="0 1 0 0 0"/>\n'
    elif robot_type == "remote_ankle_arm":
        # 사람 thigh/shank와 직접 묶지 않고 발목 위치만 cuff로 연결한다.
        _sr = 'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"'
        xml += f'    <connect name="remote_ankle_loop_crank" site1="remote_ankle_rod_end_site" site2="remote_ankle_E_site" {_sr}/>\n'
        xml += f'    <connect name="remote_ankle_loop_C" site1="remote_ankle_C_coupler_site" site2="remote_ankle_C_site" {_sr}/>\n'
        xml += f'    <tendon name="remote_ankle_parallel_long" tendon1="remote_ankle_long_links_tendon" polycoef="0 0 0 0 0" {_sr}/>\n'
        xml += f'    <tendon name="remote_ankle_parallel_short" tendon1="remote_ankle_short_links_tendon" polycoef="0 0 0 0 0" {_sr}/>\n'
        xml += f'    <connect name="remote_ankle_cuff_constraint" site1="cuff_dummy_site" site2="bracket_end_site" {_sr}/>\n'
    else:
        _sr = 'solref="0.005 1" solimp="0.95 0.99 0.001 0.5 2"'
        # motor2 crank-rocker loop.
        xml += f'    <connect name="loop_crank" site1="rod_end_site" site2="plate_E_site" {_sr}/>\n'

        # C passive hinge: coupler-end C body is connected to the C capture-link pin fixed on the EE body.
        xml += f'    <connect name="loop_distal_C" site1="distal_C_joint_site" site2="ee_C_site" {_sr}/>\n'

        # Crossed assembly 방지: 두 긴 링크는 항상 서로 평행, 두 짧은 링크도 항상 서로 평행.
        # distal_C_joint_site와 ee_C_site를 닫아 C passive joint를 만들고,
        # 아래 fixed tendon equality로 open parallelogram(| |)을 직접 강제한다.
        xml += f'    <tendon name="eq_parallel_long_links" tendon1="parallel_long_links_tendon" polycoef="0 0 0 0 0" {_sr}/>\n'
        xml += f'    <tendon name="eq_parallel_short_links" tendon1="parallel_short_links_tendon" polycoef="0 0 0 0 0" {_sr}/>\n'

        if control_mode in ["constraint_id", "robot_drives_human"]:
            xml += '    <weld name="cuff_constraint" site1="cuff_dummy_site" site2="bracket_end_site"/>\n'
        
    xml += "  </equality>\n"

    xml += "\n</mujoco>\n"
    xml = textwrap.dedent(xml)
    if bilateral:
        return _bilateralize_xml(xml, robot_type, hip_joint_spacing)
    return xml
