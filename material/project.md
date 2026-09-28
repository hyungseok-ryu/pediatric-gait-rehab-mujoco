# 소아 보행 보조 로봇팔 시뮬레이션 프로젝트 명세

> 이 문서는 프로젝트 초기에 작성된 설계 명세입니다. 현재 구현은 양측 하지,
> 외골격형 및 말단 구동형 로봇과 보행재활 UI까지 확장되었습니다. 최신 실행 방법과
> 지원 범위는 저장소 루트의 `README.md`를 기준으로 합니다.

## 1. 프로젝트 개요

본 프로젝트의 목표는 **건강인의 보행 데이터를 기반으로, 소아 환자의 하지(다리)에 부착되는 외골격형 로봇팔(robot arm)이 정상 보행 궤적을 생성·추종하도록 하는 시뮬레이션**을 MuJoCo 환경에서 구축하고, 다양한 신체 조건(키, 몸무게 등)에 대해 **모터에 요구되는 토크 프로파일**을 산출하는 것이다.

### 1.1 최종 산출물
1. 건강인 보행 데이터를 따르는 **하지(thigh + shank) MuJoCo 모델**
2. 발목(ankle)을 end-effector로 추종하는 **3-DOF 로봇팔 MuJoCo 모델**
3. **Inverse Kinematics (IK)** 솔버 → 로봇팔의 각 모터 관절 궤적 산출
4. 각 모터에 대한 **PD 제어** 적용 → **토크 프로파일** 출력
5. 소아의 신체 파라미터(키, 몸무게, 다리 길이 등)를 입력으로 받아 위 결과를 재산출하는 **파라미터화된 파이프라인**

---

## 2. 좌표계 정의

- **Global X축**: 사람의 진행 방향 (보행 방향)
- **Global Z축**: 위쪽(중력 반대 방향)
- **Global Y축**: X, Z의 오른손 좌표계 보완 축 (측면 방향)

### 운동 자유도 구분 (중요)
| 모델 | 운동 평면 | Y 방향 |
|------|----------|--------|
| 사람 하지 (thigh + shank) | **X-Z 평면(시상면)만** | Y = 0 고정 |
| 로봇팔 (robot arm) | **3차원 전체 (X, Y, Z)** | 자유 |

- 사람 다리의 ankle 위치는 X-Z 평면에서만 계산된 **2D 좌표 (x, z)** 로 출력된다.
- 로봇팔의 end-effector는 이 (x, z) 위치를 3D 공간에서 추종한다. 즉, **y 목표값은 고정값(로봇팔 base의 y 오프셋 등)** 으로 설정하거나 0으로 처리.
- 로봇팔은 3D 운동을 하므로, IK 및 동역학 계산은 **3D 기준**으로 수행한다.

---

## 3. 사람(하지) 모델

### 3.1 구조
- **2-link 하지 모델**: thigh(대퇴) + shank(하퇴)
- **Hip 관절은 Global frame에 고정** (사람 모델 자체는 공간에 매달려 있음, free-floating base 없음)
- 다리만 **공중에서** walking 동작을 수행 (지면 접촉 없음)
- 관절 자유도:
  - Hip joint: X-Z 평면 내 flexion/extension 1-DOF (Y축 회전)
  - Knee joint: X-Z 평면 내 flexion/extension 1-DOF (Y축 회전)

### 3.2 보행 데이터 입력
- 입력 파일: `gait_data.csv` (보행 데이터 csv 파일, 사용자 제공)
- **CSV 파일의 컬럼 구조는 코드에서 명시적으로 파싱 가능하도록 작성**해야 하며, 다음 중 어느 형태로든 처리 가능한 유연한 로더를 구현할 것:
  - (a) Hip angle, Knee angle 시계열 (직접 관절각 제어 가능)
  - (b) Hip / Knee / Ankle의 3D 위치 시계열 (역기구학으로 관절각 추출 필요)
- 코드 첫 부분에 **CSV 컬럼 매핑을 명시하는 설정(dict 또는 config)** 을 둘 것.
- 시간축이 데이터 내에 존재한다고 가정하며, 시뮬레이션 timestep과 보행 데이터 샘플링이 다를 경우 **보간(interpolation)** 처리한다.

### 3.3 사람 모델의 동작 방식
- 사람의 다리는 **보행 데이터를 그대로 따라가는 kinematic playback** 으로 구동된다 (즉, 사람 다리 자체는 dynamics가 아닌 position-driven으로 움직여도 무방).
- 이때 **ankle 위치의 시계열**을 로봇팔의 end-effector 추종 목표(reference trajectory)로 사용한다.

---

## 4. 로봇팔(robot arm) 모델

### 4.1 위치 및 부착
- 로봇팔의 base는 **무릎 높이 바로 위 부근**의 외부 고정점에 위치한다 (Global frame에 고정).
- 위에서 아래로 내려오며 ankle 부근까지 도달한다.
- 마지막 링크 끝은 **ankle cuff** 와 연결될 예정이나, **이번 구현에서 cuff 자체는 생략**한다. 로봇팔 end-effector의 위치가 사람 ankle의 위치를 추종하는 것으로 충분하다.

### 4.2 관절 구성 (위에서 아래 순서, 총 3-DOF)

| # | Joint 명 | 종류 | 회전축 | 비고 |
|---|---------|------|--------|------|
| 1 | base_passive | **Passive joint** | Global **X축** 회전 | 모터 없음, 자유 회전 (마찰/감쇠만 존재). 로봇팔이 Y방향으로도 흔들릴 수 있도록 허용 |
| 2 | motor1 | **Active joint** | **Y축** 회전 (시상면 내 flexion) | 토크 제어 가능 |
| 3 | (link 1) | - | - | motor1과 motor2를 잇는 강체 링크 |
| 4 | motor2 | **Active joint** | **Y축** 회전 (flexion) | 토크 제어 가능 |
| 5 | (link 2) | - | - | motor2 아래에서 ankle 부근까지 이어지는 링크 |

- **3-DOF**는 [passive X축 회전 1개] + [active Y축 회전 2개] = 3개의 회전 자유도를 의미한다.
- 로봇팔은 **3차원 공간에서 동작**하며, passive joint가 X축 회전을 담당하므로 사람 다리가 시상면(X-Z)에서만 움직이더라도 로봇팔은 Y방향 편차를 수동적으로 흡수할 수 있다.
- IK 계산은 **3D 위치 기준**으로 수행되어야 한다. (end-effector 목표: 3D 좌표 (x_target, y_target, z_target), 단 y_target는 고정 혹은 0)
- MuJoCo dynamics는 로봇팔의 3D 관성/중력 효과를 포함하므로, 토크 프로파일은 3D 기준으로 계산된다.
- **주의**: 본 작업에서는 보행이 X-Z 평면 내에서만 일어나므로 passive joint는 거의 움직이지 않지만, 모델 일반화를 위해 구조에는 반드시 포함한다.

### 4.3 링크 파라미터 (초기값, 추후 조정 가능하도록 변수화)
- 모든 링크 길이, 질량, 관성, 모터 회전축 위치는 **MuJoCo XML 상단 또는 별도 config에서 변수로 정의**하여 쉽게 바꿀 수 있도록 한다.

---

## 5. 제어 파이프라인

### 5.1 전체 흐름
```
[gait_data.csv]
    ↓
[사람 하지 kinematic playback]
    ↓
[ankle 위치 시계열 → 로봇팔 end-effector reference trajectory]
    ↓
[Inverse Kinematics (로봇팔 3-DOF)] 
    ↓
[motor1, motor2의 목표 관절각(θ_des) 시계열]   ※ passive joint는 자유
    ↓
[PD 제어 (각 active 모터)]
    ↓
[모터 토크 프로파일 τ(t) 출력 및 저장]
```

### 5.2 Inverse Kinematics 요구사항
- 로봇팔의 end-effector가 ankle 목표 위치를 정확히 추종하도록 풀 것.
- **IK는 3D 공간 기준**으로 수행한다. 입력: 3D 목표 위치 (x, y, z), 출력: motor1 각도, motor2 각도.
  - passive joint(X축 회전)는 IK 변수에 포함하지 않고, 시뮬레이션 중 자연스럽게 따라오도록 둔다.
  - 사람 ankle의 y 좌표는 0(또는 고정값)이므로 IK 목표의 y는 고정이지만, **3D FK 기반으로 계산해야 3D 동역학과 일치**한다.
- IK는 **analytical solution이 가능하면 우선 사용**, 어려우면 numerical (Jacobian pseudo-inverse 또는 MuJoCo의 `mj_jac` 활용한 반복법)로 구현.
- IK 해가 여러 개일 경우(elbow up/down 등), **무릎 굽힘 방향을 사람의 무릎 굽힘 방향과 일치**시키는 분기를 선택할 것.

### 5.3 PD 제어
- 각 active 모터에 대해 다음 형태로 토크 계산:
  ```
  τ_i(t) = Kp_i * (θ_des_i(t) - θ_i(t)) + Kd_i * (θ̇_des_i(t) - θ̇_i(t))
  ```
- Kp, Kd 게인은 모터별로 별도 설정 가능하도록 파라미터화.
- 게인 초기값은 합리적인 디폴트를 두되, 코드 상단에서 쉽게 튜닝 가능하도록 노출.

---

## 6. 환자 파라미터화 (소아 대응)

다양한 소아(또는 일반 사용자)의 신체 조건에 대해 토크를 산출할 수 있도록, 다음 입력값을 **상단 config 형태로 노출**할 것:

| 파라미터 | 단위 | 설명 |
|----------|------|------|
| `subject_height` | m | 대상자 키 |
| `subject_weight` | kg | 대상자 체중 |
| `thigh_length` | m | 대퇴 길이 (미입력 시 키 기반 추정) |
| `shank_length` | m | 하퇴 길이 (미입력 시 키 기반 추정) |
| `thigh_mass` | kg | 대퇴 질량 (미입력 시 체중 비율로 추정) |
| `shank_mass` | kg | 하퇴 질량 (미입력 시 체중 비율로 추정) |
| `foot_mass` | kg | 발 질량 (로봇팔이 들어 올려야 할 말단 부하 포함 여부 명시) |

### 6.1 추정식 (입력 누락 시 기본값)
- 인체 분절 파라미터 기본 추정에는 일반적으로 사용되는 Winter의 인체측정학 비율을 사용한다 (예: thigh length ≈ 0.245 × height, shank length ≈ 0.246 × height, thigh mass ≈ 0.100 × body mass, shank mass ≈ 0.0465 × body mass 등).
- 사용된 추정식은 코드 주석으로 출처 표기 + 한 곳에 모아둘 것 (`anthropometry.py` 또는 동일 역할의 섹션).

### 6.2 보행 데이터의 스케일링
- 보행 데이터(`gait_data.csv`)가 특정 신체 조건에서 얻어진 것일 수 있으므로, 키/다리 길이에 따라 ankle 궤적을 **선형 스케일링** 하는 옵션을 둘 것. (스케일링 on/off 가능)

---

## 7. MuJoCo 환경 사양

- **MuJoCo 버전: 3.3.0** 사용 (반드시 3.3.0과 호환되는 API 사용).
- Python 바인딩: `mujoco` 패키지 (공식 Google DeepMind 패키지). `dm_control` 의존성은 두지 말 것.
- 시뮬레이션 timestep, integrator 등은 XML 상단에서 명시.
- 가능한 한 **하나의 MuJoCo XML**에 사람 하지 + 로봇팔을 함께 정의하되, body group / visual group을 분리하여 디버깅이 쉽도록 한다.
- 중력은 활성화 (`-Z` 방향). 단, 사람 다리가 kinematic playback일 경우 사람 모델의 dynamics는 무시될 수 있고 (mocap body 등 활용 가능), **로봇팔은 반드시 dynamics가 살아 있어야 한다** (토크 계산이 목적이므로).

---

## 8. 출력물(파일/플롯) 요구사항

코드 실행 후 다음을 출력/저장할 것:

1. **시뮬레이션 영상 또는 실시간 viewer 옵션**
   - `mujoco.viewer` 기반 실시간 표시 옵션
   - off-screen 렌더링으로 mp4 저장 옵션
2. **수치 데이터 (CSV)**
   - 시간, 사람 ankle 목표 위치 (x, z)
   - 로봇팔 end-effector 실제 위치 (x, z)
   - motor1, motor2 목표각 / 실제각 / 각속도
   - motor1, motor2 토크 τ(t)
   - passive joint 각도(참고용)
3. **플롯(matplotlib)**
   - End-effector tracking error (x, z) vs time
   - 각 모터 관절각 (target vs actual) vs time
   - 각 모터 토크 vs time
   - (옵션) 각 모터의 토크-속도 곡선

---

## 9. 코드 구조 권장사항

```
project/
├── models/
│   └── leg_and_arm.xml          # 사람 하지 + 로봇팔 통합 MuJoCo 모델
├── data/
│   └── gait_data.csv            # 입력 보행 데이터
├── src/
│   ├── config.py                # 환자 파라미터, 게인, 경로 등 모든 설정
│   ├── anthropometry.py         # 신체 비율 추정
│   ├── gait_loader.py           # CSV 파싱 + 보간
│   ├── kinematics.py            # 사람 다리 FK, 로봇팔 FK/IK
│   ├── controller.py            # PD 제어
│   ├── simulate.py              # MuJoCo 메인 루프
│   └── plot_results.py          # 결과 시각화
└── outputs/
    ├── torque_profile.csv
    ├── tracking.csv
    └── plots/
```

- 각 모듈은 단독으로 import 가능해야 하며, **`simulate.py` 단일 실행으로 전체 파이프라인이 돌아야 한다**.

---

## 10. 우선순위 및 단계적 구현 (요청)

다음 순서로 단계적으로 구현해주세요. **각 단계가 끝날 때마다 동작 확인이 가능한 형태**여야 합니다.

1. **Step 1**: MuJoCo XML로 사람 2-link 하지 모델 작성 + viewer로 가시화 (정지 상태)
2. **Step 2**: `gait_data.csv` 로딩 + 사람 하지에 보행 동작 playback
3. **Step 3**: 로봇팔 3-DOF 모델 추가 (사람 모델 위쪽 base에 부착) + 가시화
4. **Step 4**: ankle 위치 시계열 추출 → 로봇팔 IK 풀이 → 모터 목표각 산출
5. **Step 5**: 로봇팔에 PD 제어 적용 → 토크로 ankle 추종
6. **Step 6**: 토크 프로파일 및 tracking error 저장/플롯
7. **Step 7**: 환자 파라미터(키, 몸무게 등)에 따른 재실행 자동화 + 비교 플롯

---

## 11. 불명확/확인 필요 사항 (AI는 작성 전에 이 부분에 대해 가정한 바를 명시하고 진행할 것)

다음 항목들은 명세가 모호하므로, 코드 작성 시 **합리적인 기본값을 가정하고, 그 가정을 코드 주석과 README에 명확히 남겨주세요**:

- [ ] `gait_data.csv`의 정확한 컬럼 스키마 (시간 단위, 각도 단위(deg/rad), 좌표 단위(m/mm))
- [ ] 로봇팔 base의 정확한 위치 좌표 (무릎 높이 위 어느 정도 위인가?)
- [ ] 로봇팔 link 길이 (사람 다리 길이 대비 비율로 잡을지)
- [ ] 로봇팔 링크/모터의 질량, 관성 (모터 사양 미정 시 합리적 추정치)
- [ ] PD 게인 초기값 (코드 상에서 쉽게 튜닝 가능하도록)
- [ ] 발(foot)의 질량을 ankle 추종 부하에 포함시킬지 여부
- [ ] 로봇팔 모터의 토크 한계 (saturation 적용 여부)

---

## 12. 코딩 스타일/품질

- Python 3.10+ 기준.
- 타입 힌트 사용.
- 함수/클래스 단위에 간결한 docstring.
- 핵심 수식(IK, PD, 인체 비율 추정 등)에는 **수식 자체를 주석으로 명시**.
- `requirements.txt` 포함: `mujoco==3.3.0`, `numpy`, `scipy`, `pandas`, `matplotlib`.

---

## 13. 최종 요청 사항

위 명세에 따라:
1. 위 디렉토리 구조대로 코드를 작성하고,
2. 각 파일의 전체 내용을 제공하며,
3. 실행 방법(`python src/simulate.py` 등)과 의존성 설치 방법을 README에 정리해주세요.
4. 불명확한 부분은 임의 가정 후 **README의 "Assumptions" 섹션에 모두 기록**해주세요.
