"""3–4세 소아의 Comfortable 오른쪽 hip/knee 평균 보행 궤적을 추출한다.

입력 파일의 ``3-4 Cycle Waves`` 시트에서 네 대상자의 Angle 파형을
보행주기별로 평균한다. 원본 Excel은 수정하지 않으며, 시뮬레이션에서 바로
읽을 수 있는 CSV와 추출 조건 메타데이터를 생성한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "sagittal_cycle_metrics.xlsx"
DEFAULT_OUTPUT = ROOT / "data" / "gait_3to4_comfortable_right_mean.csv"


def extract_mean_gait(input_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    waves = pd.read_excel(input_path, sheet_name="3-4 Cycle Waves")
    gait_columns = [column for column in waves.columns if str(column).startswith("GC ")]
    if len(gait_columns) != 100:
        raise ValueError(f"Expected 100 gait-cycle columns, found {len(gait_columns)}")

    selection = waves[
        (waves["Condition"] == "Comfortable")
        & (waves["Side"] == "Right")
        & (waves["Joint"].isin(["Hip", "Knee"]))
        & (waves["Variable (unit)"] == "Angle (deg)")
    ].copy()
    counts = selection.groupby("Joint")["Subject ID"].nunique().to_dict()
    if counts != {"Hip": 4, "Knee": 4}:
        raise ValueError(f"Unexpected selected subject counts: {counts}")

    hip = selection[selection["Joint"] == "Hip"]
    knee = selection[selection["Joint"] == "Knee"]
    hip_mean = hip[gait_columns].astype(float).mean(axis=0).to_numpy()
    knee_mean = knee[gait_columns].astype(float).mean(axis=0).to_numpy()
    hip_sd = hip[gait_columns].astype(float).std(axis=0, ddof=1).to_numpy()
    knee_sd = knee[gait_columns].astype(float).std(axis=0, ddof=1).to_numpy()

    subject_stride = (
        selection[["Subject ID", "Stride time (s)"]]
        .drop_duplicates()
        .sort_values("Subject ID")
    )
    cycle_duration = float(subject_stride["Stride time (s)"].mean())
    phase = np.linspace(0.0, 100.0, len(gait_columns))
    time = cycle_duration * phase / 100.0

    output = pd.DataFrame(
        {
            "Subject": "Age3to4_Comfortable_Right_Mean",
            "Condition": "Comfortable",
            "Side": "Right",
            "N": 4,
            "MeanAge_years": 3.75,
            "SourceMeanHeight_m": 1.01175,
            "SourceMeanWeight_kg": 16.2,
            "CycleDuration_s": cycle_duration,
            "Phase": phase,
            "time": time,
            "hip_angle": hip_mean,
            "knee_angle": knee_mean,
            "hip_angle_sd": hip_sd,
            "knee_angle_sd": knee_sd,
        }
    )
    metadata: dict[str, object] = {
        "source_workbook": input_path.name,
        "source_sheet": "3-4 Cycle Waves",
        "condition": "Comfortable",
        "side": "Right",
        "joints": ["Hip", "Knee"],
        "variable": "Angle (deg)",
        "subject_ids": sorted(selection["Subject ID"].unique().tolist()),
        "n_subjects": 4,
        "age_composition": {"3 years": 1, "4 years": 3},
        "mean_age_years": 3.75,
        "source_mean_height_m": 1.01175,
        "source_mean_weight_kg": 16.2,
        "mean_stride_time_s": cycle_duration,
        "individual_stride_times_s": {
            str(row["Subject ID"]): float(row["Stride time (s)"])
            for _, row in subject_stride.iterrows()
        },
        "n_gait_points": len(gait_columns),
        "hip_min_deg": float(np.min(hip_mean)),
        "hip_max_deg": float(np.max(hip_mean)),
        "knee_min_deg": float(np.min(knee_mean)),
        "knee_max_deg": float(np.max(knee_mean)),
    }
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output, metadata = extract_mean_gait(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.9f")
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved mean gait: {args.output}")
    print(f"Saved metadata: {metadata_path}")
    print(
        f"N={metadata['n_subjects']}, stride={metadata['mean_stride_time_s']:.4f}s, "
        f"hip={metadata['hip_min_deg']:.2f}..{metadata['hip_max_deg']:.2f}deg, "
        f"knee={metadata['knee_min_deg']:.2f}..{metadata['knee_max_deg']:.2f}deg"
    )


if __name__ == "__main__":
    main()
