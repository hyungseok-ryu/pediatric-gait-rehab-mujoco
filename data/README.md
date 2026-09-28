# 보행 데이터

이 디렉터리에는 시뮬레이션을 재현하는 데 필요한 입력 데이터만 저장합니다.

| 파일 | 용도 |
| --- | --- |
| `gait_3to4_comfortable_right_mean.csv` | 기본 보행 입력. 3–4세 네 대상자의 Comfortable/Right 고관절·무릎 평균 1주기 |
| `gait_3to4_comfortable_right_mean.json` | 기본 입력의 추출 조건, 대상자 수, 주기와 관절각 범위 |
| `sagittal_cycle_metrics.xlsx` | 평균 CSV를 생성한 원본 workbook. 재배포 권한 확인 전에는 공개 저장소에서 제외 |
| `gait_data.csv` | 기존 대상자·속도별 배치 분석용 입력 |

## 기본 CSV 단위

- `time`: 초
- `Phase`: 보행 주기 백분율
- `hip_angle`, `knee_angle`: 도(degree), 굴곡이 양수
- 시뮬레이션 내부 관절각: radian으로 변환

기본 파일은 오른쪽 파형만 제공합니다. 왼쪽은 코드에서 같은 파형을 보행 주기의
50%만큼 앞당겨 생성하므로 좌우 대칭 교대보행을 가정합니다.

## 기본 평균 파형 재생성

원본 workbook을 적법하게 보유한 경우 `data/sagittal_cycle_metrics.xlsx`에 배치한
뒤 다음 명령을 실행합니다. 이 파일은 기본적으로 Git에서 제외됩니다.

```bash
python src/extract_pediatric_mean_gait.py
```

다른 입출력 경로를 지정할 수도 있습니다.

```bash
python src/extract_pediatric_mean_gait.py \
  --input data/sagittal_cycle_metrics.xlsx \
  --output data/gait_3to4_comfortable_right_mean.csv
```

## 공개 전 확인

`sagittal_cycle_metrics.xlsx`를 공개 GitHub 저장소에 포함하려면 원 데이터의
라이선스와 재배포 조건을 먼저 확인해야 합니다. 재배포가 허용되지 않는 경우 원본
workbook은 Git에서 제외하고, 공개 가능한 파생 CSV와 메타데이터만 배포하십시오.
