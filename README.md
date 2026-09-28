# Pediatric Gait Rehabilitation MuJoCo Simulation

소아 보행재활 로봇을 위한 양측 하지 MuJoCo 시뮬레이션입니다. 소아용 수동보행기
프레임에 장착된 외골격형(`exoskeleton`)과 말단 구동형(`open_chain`) 로봇을
비교할 수 있으며, 보행 궤적 추종, 관절별 보조율, 기립 유지, 안전 정지 및
실시간 재활 제어 UI를 제공합니다.

> 이 저장소는 연구용 시뮬레이션입니다. 실제 환자 또는 의료기기 제어에 바로
> 사용해서는 안 되며, 환자별 관절가동범위와 제어 파라미터는 임상 전문가의
> 평가와 별도의 안전 검증이 필요합니다.

## 주요 기능

- 오른쪽·왼쪽 하지와 로봇을 모두 포함한 양측 모델
- 고관절·무릎 동축 외골격형과 발목 추종 말단 구동형 구조
- 3–4세 소아 평균 보행각 기반의 교대보행
- 좌우 고관절·무릎 구동 범위와 보조율의 독립 설정
- 기립 유지, 로봇 주도 보행, 능동 보조 및 안전 정지 모드
- 안전 정지 또는 기립 상태에서 보행으로 전환할 때 S-curve 부드러운 시작
- MuJoCo viewer를 사용하는 한글 데스크톱 UI
- 단일 실행, 배치 실행 및 open-chain 링크 길이 sweep
- 관절각, 발목 추종, 모터 토크와 세션 설정 CSV 기록

## 저장소 구성

```text
pediatric-gait-rehab-mujoco/
├── .github/workflows/        # GitHub Actions headless smoke test
├── data/                     # 입력 보행 데이터와 데이터 설명
├── material/                 # 설계·제어 문서
├── models/                   # 가장 최근에 생성된 MuJoCo XML 예시
├── src/
│   ├── config.py             # 체형, 로봇, 제어 게인과 공통 경로
│   ├── anthropometry.py      # 소아 신체 분절 추정
│   ├── gait_loader.py        # 보행 CSV 로딩과 보간
│   ├── kinematics.py         # 하지 FK와 로봇 IK
│   ├── controller.py         # PD 제어기
│   ├── build_xml.py          # 양측 MuJoCo XML 생성
│   ├── simulate.py           # 메인 시뮬레이션과 CLI
│   ├── rehab_ui.py           # 보행재활 제어 UI
│   ├── run_batch.py          # 배치 시뮬레이션
│   ├── extract_pediatric_mean_gait.py
│   └── sweep_open_chain_extensions.py
├── .gitignore
├── requirements.txt
└── README.md
```

`outputs/`, `MUJOCO_LOG.TXT`, 가상환경, Python 캐시와 IDE 설정은 실행에 필요하지
않으므로 Git에서 제외됩니다. 시뮬레이션을 실행하면 `outputs/`는 자동 생성됩니다.

## 요구 환경

- Python 3.10 또는 3.11 권장
- MuJoCo 3.3.0
- 실시간 UI 사용 시 Tk
- viewer 또는 영상 렌더링 사용 시 OpenGL

현재 고정된 패키지 조합은 Ubuntu 22.04.5, Python 3.10.12에서 검증했습니다.
수치 시뮬레이션은 운영체제와 무관하게 같은 명령으로 실행되며, viewer 준비 방법만
운영체제별로 다를 수 있습니다.

## 설치

저장소를 복제한 뒤 모든 명령은 저장소 최상위 디렉터리에서 실행합니다.

```bash
git clone https://github.com/hyungseok-ryu/pediatric-gait-rehab-mujoco.git
cd pediatric-gait-rehab-mujoco
```

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y python3-venv python3-tk libgl1 libegl1 libglfw3

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

Python 3.10 또는 3.11을 먼저 설치한 후 다음을 실행합니다. 일반적인 Windows용
Python 설치에는 Tk가 포함됩니다.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS에서 MuJoCo의 수동 viewer를 열 때는 일반 `python` 대신 MuJoCo가 설치하는
`mjpython` 실행기가 필요할 수 있습니다.

## 설치 확인

그래픽 창을 사용하지 않는 1주기 smoke test입니다.

```bash
python src/simulate.py --render none --cycles 1 --outdir smoke_test
```

성공하면 다음 파일이 생성됩니다.

```text
outputs/smoke_test/torque_profile.csv
```

명령행 옵션을 확인하려면 다음을 실행합니다.

```bash
python src/simulate.py --help
```

## 보행재활 UI 실행

```bash
python src/rehab_ui.py
```

macOS에서 viewer가 열리지 않으면 다음 명령을 사용합니다.

```bash
mjpython src/rehab_ui.py
```

UI에서 세션 시작 전에 체형, 로봇 구조와 좌우 관절 구동 범위를 설정합니다.
기본 관절 범위는 현재 입력 데이터에 여유를 둔 고관절 `-15~50°`, 무릎
`0~80°`입니다. 실행 중에는 보행 모드, 보행 속도, 부드러운 시작 시간,
좌우 관절 보조율과 가상 환자 움직임 입력을 조절할 수 있습니다.

관절 범위는 다음 두 곳에 함께 적용됩니다.

1. 입력 보행 목표 궤적의 최소·최대각 제한
2. MuJoCo 사람 관절과 외골격형 동축 관절의 물리적 hard limit

기립·보행 전환을 위해 설정 범위에는 고관절 `0°`, 무릎 `5°`가 포함되어야
합니다. 각 세션의 범위와 제어 설정은 `torque_profile.csv`에도 저장됩니다.

## 명령행 실행 예시

실시간 viewer:

```bash
python src/simulate.py --render viewer --cycles 5
```

그래픽 없이 계산만 수행:

```bash
python src/simulate.py --render none --cycles 5
```

로봇 구조 선택:

```bash
python src/simulate.py --render viewer --robot_type exoskeleton
python src/simulate.py --render viewer --robot_type open_chain
```

체형과 골반 폭 지정:

```bash
python src/simulate.py --render none --height 0.856 --weight 12.0
python src/simulate.py --render none --hip-spacing 0.14
```

open-chain 링크 길이를 수동 지정할 때 단위는 m입니다.

```bash
python src/simulate.py --render none --robot_type open_chain \
  --link1 0.25 --link2 0.25
```

기존 속도별 보행 데이터에서 대상자와 속도 선택:

```bash
python src/simulate.py --render none \
  --gait-data data/gait_data.csv --subject Sub06 --speed 0.912
```

영상 저장:

```bash
python src/simulate.py --render offscreen --cycles 1
python src/simulate.py --render viewer --cycles 1 --save-video
python src/simulate.py --render offscreen --video-path outputs/demo.mp4
```

디스플레이가 없는 Linux 서버에서는 EGL을 지정합니다.

```bash
MUJOCO_GL=egl python src/simulate.py --render offscreen --cycles 1
```

서버에 EGL이 없다면 영상 없이 `--render none`을 사용합니다.

## 배치 시뮬레이션

실행 조건을 먼저 확인합니다.

```bash
python src/run_batch.py --dry-run
```

일부 조건만 실행하거나 병렬 실행할 수 있습니다.

```bash
python src/run_batch.py --render none --cycles 2 --max-runs 20
python src/run_batch.py --render none --cycles 2 --workers 4
python src/run_batch.py --render none --cycles 2 --skip-existing
```

배치 결과는 `outputs/batch_results/`, 요약은 `outputs/batch_summary.csv`에
저장됩니다.

## Open-chain 링크 길이 sweep

기본값은 체형 기준 링크 길이에서 각 링크를 최대 100 mm까지 20 mm 간격으로
연장하여 비교합니다. 영상은 생성하지 않습니다.

```bash
python src/sweep_open_chain_extensions.py
```

빠른 기능 확인용 최소 sweep:

```bash
python src/sweep_open_chain_extensions.py \
  --max-extension-mm 0 --cycles 1 --force
```

## 보행 데이터

기본 입력은 `data/gait_3to4_comfortable_right_mean.csv`입니다. 오른쪽 고관절과
무릎의 1주기 평균 파형이며, 왼쪽은 같은 파형을 50% 위상 이동하여 생성합니다.
따라서 현재 기본 모델은 좌우 대칭 교대보행을 가정합니다.

원본 파형의 평균 보행 주기는 0.7775초이고, 기본 만 2세 시뮬레이션에서는 파형을
유지하면서 1.0초로 시간 스케일링합니다. 원본 Excel workbook은 재배포 권한이
확인되지 않아 공개 저장소에서 제외하며, 재현에 필요한 파생 CSV와 메타데이터는
포함합니다. 데이터 파일별 설명은 [data/README.md](data/README.md)를 참고하십시오.

## 주요 설정

대부분의 기본값은 `src/config.py`에서 관리합니다.

| 설정 | 내용 |
| --- | --- |
| `SubjectParams` | 키, 체중, 하지 길이·질량과 좌우 hip 간격 |
| `RobotArmParams` | 로봇 구조, 링크 길이·질량과 토크 한계 |
| `WalkerParams` | 고정식 소아 보행기 시각 모델 |
| `PDGains` | Motor 1/2의 PD 게인 |
| `CSV_COLUMN_MAP` | 보행 CSV의 시간·관절각 컬럼 |
| `SIM_TIMESTEP` | MuJoCo 시뮬레이션 시간 간격 |

모델 XML은 실행할 때 선택한 체형과 로봇 구조에 맞춰 다시 생성되며
`models/leg_and_arm.xml`을 덮어씁니다. 이 파일은 최근 생성 모델을 확인하기 위한
예시이므로 영구 설정값은 `src/config.py`와 실행 인자에서 변경하십시오.

## 문서

- [제어 아키텍처와 관절 범위](material/control_architecture.md)
- [말단 구동형 구조 설명](material/end_effector_robot_arm_concept.md)
- [초기 프로젝트 설계 명세](material/project.md)
- [배치 시뮬레이션 메모](material/run_batch_simulation.md)

## 결과 파일과 Git 관리

실행 결과, 로그와 영상은 모두 로컬 `outputs/`에 저장됩니다. 이 디렉터리와
`MUJOCO_LOG.TXT`는 `.gitignore`에 포함되어 GitHub에 올라가지 않습니다.

현재 공개 구성에는 라이선스 파일이 없습니다. 따라서 저장소를 열람하고 복제할
수는 있지만, 별도 허가 없이 코드를 재사용·수정·배포할 권한은 부여되지 않습니다.
추후 라이선스를 선택하면 `LICENSE` 파일을 추가하십시오.

`data/sagittal_cycle_metrics.xlsx` 원본은 라이선스와 재배포 조건이 확인될 때까지
`.gitignore`로 제외합니다. 허가를 확인한 경우에만 공개 저장소에 추가하십시오.
