# run_batch.py 배치 시뮬레이션 요약

> 아래 내용은 `remote_parallelogram` 구조를 사용했던 이전 대규모 실험 기록입니다.
> 현재 `run_batch.py`의 활성 구조와 조건은 코드 및 저장소 루트의 `README.md`를
> 기준으로 확인하십시오.

## 현재 진행 상태

- 전체 조건 수: 311,040개
- 현재 진행: 약 65,000개
- 아직 전체 결과는 완료되지 않았으며, `outputs/batch_summary.csv`에 완료된 run부터 누적 저장되고 있다.

## 실행 방식

`run_batch.py`는 각 조건 조합마다 `src/simulate.py`를 별도 프로세스로 실행한다. 각 run에서는 선택된 보행 데이터, 신체 조건, 로봇팔 길이 조건, PD gain을 `simulate.py` 인자로 넘기고, 결과를 개별 출력 폴더의 `torque_profile.csv`로 저장한다.

현재 활성화된 로봇 구조는 `remote_parallelogram`이다.

## PD Gain 조건

현재 배치에서는 두 모터에 같은 PD gain을 사용한다.

| Motor | Kp | Kd |
|---|---:|---:|
| motor1 | 1500.0 | 15.0 |
| motor2 | 1500.0 | 15.0 |

조건 이름은 `M1Kp1500Kd15_M2Kp1500Kd15`로 기록된다.

## 신체 조건

신체 조건은 3개 age case로 sweep한다.

| 조건 | 키 [m] | 체중 [kg] |
|---|---:|---:|
| 5yo | 1.10 | 20.0 |
| 6yo | 1.18 | 22.0 |
| 7yo | 1.25 | 25.0 |

허벅지, 종아리, 발의 분절 길이와 질량은 `anthropometry.py`에서 Winter 비율을 사용해 추정된다.

## 길이 조건

`remote_parallelogram` 구조의 길이 조건은 아래 7개 파라미터 조합으로 구성된다.

| 파라미터 | 의미 | 값 [m] |
|---|---|---|
| A | motor1 -> motor2 수평 거리 | 0.11, 0.13, 0.15, 0.17 |
| B | motor1 -> motor2 수직 거리 | 0.02, 0.04, 0.06, 0.08 |
| C | motor2 crank/rod 공통 길이 | 0.07, 0.09, 0.11, 0.13 |
| D | AE/BE upper plate link 길이 | 0.06, 0.07, 0.08, 0.09 |
| E | AB/CD short spacing | 0.07, 0.09, 0.11, 0.13 |
| F | AD/BC drive link 길이 | 0.20, 0.25, 0.30, 0.35, 0.40, 0.45 |
| G | distal link2 길이 | 0.20, 0.25, 0.30, 0.35, 0.40, 0.45 |

기하학적으로 `D >= E / 2`를 만족하지 않는 조합은 생성하지 않는다. 이 필터를 적용한 뒤 remote arm 조건은 34,560개가 된다.

## 전체 조건 수

현재 총 311,040개 조건은 다음 조합으로 구성된다.

```text
3 gait cases
× 3 신체 조건
× 34,560 remote arm 길이 조건
× 1 PD gain 조건
= 311,040 runs
```

## 통과 조건

각 run은 다음 기준으로 상태가 결정된다.

| status | 의미 |
|---|---|
| `success` | `simulate.py`가 정상 종료되고, 추종 오차 기준을 통과한 경우 |
| `failed_ik` | 사전 IK 검증에서 목표 ankle trajectory 중 하나라도 도달 불가능한 경우 |
| `failed_dynamic` | 시뮬레이션 중 0.5초 이후 순간 ankle 추종 오차가 60 mm를 초과한 경우 |
| `failed_tracking` | 정상 종료했지만 95 percentile tracking error가 10 mm를 초과한 경우 |
| `failed_error` | 그 외 비정상 종료 |
| `timeout` / `exception` | 제한 시간 초과 또는 실행 중 예외 |

최종적으로 통과로 보는 조건은 `status == success`이다.
