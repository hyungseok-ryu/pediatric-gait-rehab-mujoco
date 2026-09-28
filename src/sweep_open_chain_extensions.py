"""EXO 정렬 길이에서 각 링크를 최대 100 mm 연장하며 open-chain 성능을 sweep한다."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import config as cfg


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAIT_CSV = ROOT / "data" / "gait_3to4_comfortable_right_mean.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "open_chain_extension_sweep"


def compute_metrics(
    csv_path: Path,
    n_cycles: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """마지막 보행 주기의 추종 오차와 모터 토크 지표를 계산한다."""
    if n_cycles < 1:
        raise ValueError("n_cycles는 1 이상이어야 합니다.")
    data = pd.read_csv(csv_path)
    samples_per_cycle = len(data) // n_cycles
    if samples_per_cycle < 2:
        raise ValueError(
            f"마지막 보행 주기를 계산하기에 샘플이 부족합니다: {csv_path}"
        )
    cycle = data.tail(samples_per_cycle).copy()

    tracking_error_mm = 1000.0 * np.hypot(
        cycle["ankle_actual_x"] - cycle["ankle_target_x"],
        cycle["ankle_actual_z"] - cycle["ankle_target_z"],
    )
    motor1_torque = cycle["motor1_torque"].to_numpy(dtype=float)
    motor2_torque = cycle["motor2_torque"].to_numpy(dtype=float)
    combined_torque = np.hypot(motor1_torque, motor2_torque)
    dt = float(np.mean(np.diff(cycle["time"].to_numpy(dtype=float))))
    absolute_power = (
        np.abs(motor1_torque * cycle["motor1_thetadot"].to_numpy(dtype=float))
        + np.abs(
            motor2_torque * cycle["motor2_thetadot"].to_numpy(dtype=float)
        )
    )

    metrics = {
        "n_samples": float(len(cycle)),
        "tracking_rmse_mm": float(np.sqrt(np.mean(tracking_error_mm**2))),
        "tracking_p95_mm": float(np.percentile(tracking_error_mm, 95.0)),
        "tracking_max_mm": float(np.max(tracking_error_mm)),
        "motor1_peak_nm": float(np.max(np.abs(motor1_torque))),
        "motor2_peak_nm": float(np.max(np.abs(motor2_torque))),
        "motor1_rms_nm": float(np.sqrt(np.mean(motor1_torque**2))),
        "motor2_rms_nm": float(np.sqrt(np.mean(motor2_torque**2))),
        "combined_peak_nm": float(np.max(combined_torque)),
        "combined_rms_nm": float(np.sqrt(np.mean(combined_torque**2))),
        "absolute_mechanical_work_j": float(np.sum(absolute_power) * dt),
    }
    return metrics, cycle


def _run_case(
    *,
    robot_type: str,
    gait_csv: Path,
    output_dir: Path,
    height: float,
    weight: float,
    link1: float,
    link2: float,
    cycles: int,
    cycle_duration: float,
    case_name: str,
    force: bool,
) -> Path:
    case_dir = output_dir / "cases" / case_name
    csv_path = case_dir / "torque_profile.csv"
    metadata_path = case_dir / "sweep_input.json"
    requested = {
        "robot_type": robot_type,
        "gait_csv": str(gait_csv),
        "height_m": height,
        "weight_kg": weight,
        "link1_m": link1,
        "link2_m": link2,
        "cycles": cycles,
        "cycle_duration_s": cycle_duration,
    }
    if csv_path.exists() and metadata_path.exists() and not force:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous == requested:
            print(f"[reuse] {case_name}")
            return csv_path

    relative_outdir = case_dir.relative_to(cfg.PATHS["output_dir"])
    command = [
        sys.executable,
        str(ROOT / "src" / "simulate.py"),
        "--render",
        "none",
        "--cycles",
        str(cycles),
        "--height",
        str(height),
        "--weight",
        str(weight),
        "--robot_type",
        robot_type,
        "--gait-data",
        str(gait_csv),
        "--cycle-duration",
        str(cycle_duration),
        "--outdir",
        str(relative_outdir),
    ]
    if robot_type == "open_chain":
        command.extend(["--link1", str(link1), "--link2", str(link2)])

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "simulation.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0 or not csv_path.exists():
        raise RuntimeError(
            f"Simulation failed for {case_name} (exit={result.returncode}). "
            f"See {case_dir / 'simulation.log'}"
        )
    metadata_path.write_text(
        json.dumps(requested, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {case_name}")
    return csv_path


def _metric_grid(
    results: pd.DataFrame,
    extension_values: np.ndarray,
    metric: str,
) -> np.ndarray:
    pivot = results.pivot(index="link1_extension_mm", columns="link2_extension_mm", values=metric)
    return pivot.reindex(index=extension_values, columns=extension_values).to_numpy(dtype=float)


def _annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    lengths1_mm: np.ndarray,
    lengths2_mm: np.ndarray,
    title: str,
    fmt: str,
) -> None:
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmin(finite) + np.nanmax(finite)) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                color = "white" if value < midpoint else "black"
                ax.text(column, row, format(value, fmt), ha="center", va="center", fontsize=8, color=color)

    minimum = np.unravel_index(np.nanargmin(values), values.shape)
    ax.scatter(
        minimum[1] + 0.31, minimum[0] + 0.31, marker="*", s=75, facecolor="#ffdd57",
        edgecolor="black", linewidth=0.8, zorder=4,
    )
    ax.add_patch(Rectangle((-0.48, -0.48), 0.96, 0.96, fill=False, edgecolor="white", linewidth=2.2))
    ax.set_xticks(np.arange(len(lengths2_mm)), [f"{value:.1f}" for value in lengths2_mm])
    ax.set_yticks(np.arange(len(lengths1_mm)), [f"{value:.1f}" for value in lengths1_mm])
    ax.set_xlabel("Link 2 length (mm)")
    ax.set_ylabel("Link 1 length (mm)")
    ax.set_title(title, fontweight="bold")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_heatmaps(
    results: pd.DataFrame,
    exo: pd.Series,
    extension_values: np.ndarray,
    nominal_link1_mm: float,
    nominal_link2_mm: float,
    output_dir: Path,
    height: float,
    weight: float,
) -> None:
    lengths1 = nominal_link1_mm + extension_values
    lengths2 = nominal_link2_mm + extension_values
    panels = [
        ("tracking_rmse_mm", "Ankle tracking RMSE (mm)", ".2f", exo["tracking_rmse_mm"]),
        ("combined_rms_nm", "Combined RMS torque (Nm)", ".2f", exo["combined_rms_nm"]),
        ("motor1_rms_nm", "Motor 1 RMS torque (Nm)", ".2f", exo["motor1_rms_nm"]),
        ("motor2_rms_nm", "Motor 2 RMS torque (Nm)", ".2f", exo["motor2_rms_nm"]),
        ("motor1_peak_nm", "Motor 1 peak torque (Nm)", ".2f", exo["motor1_peak_nm"]),
        ("motor2_peak_nm", "Motor 2 peak torque (Nm)", ".2f", exo["motor2_peak_nm"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2))
    for ax, (metric, title, fmt, exo_value) in zip(axes.flat, panels):
        values = _metric_grid(results, extension_values, metric)
        _annotated_heatmap(
            ax,
            values,
            lengths1,
            lengths2,
            f"{title}\nEXO reference: {exo_value:.2f}",
            fmt,
        )
    fig.suptitle(
        f"Open-chain link-length sweep — height={height * 100:.1f} cm, weight={weight:.1f} kg",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Sweep bounds: EXO-aligned length to +100 mm per link. "
        "White box = zero extension; small star = minimum in each panel; 1.00 s/gait cycle.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(output_dir / "open_chain_link_sweep_heatmaps.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "open_chain_link_sweep_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.ones(len(x), dtype=bool)
    for index in range(len(x)):
        dominated = (
            (x <= x[index])
            & (y <= y[index])
            & ((x < x[index]) | (y < y[index]))
        )
        if np.any(dominated):
            mask[index] = False
    return mask


def save_pareto_figure(results: pd.DataFrame, exo: pd.Series, output_dir: Path) -> None:
    x = results["tracking_rmse_mm"].to_numpy(dtype=float)
    y = results["combined_rms_nm"].to_numpy(dtype=float)
    total_extension = (
        results["link1_extension_mm"].to_numpy(dtype=float)
        + results["link2_extension_mm"].to_numpy(dtype=float)
    )
    pareto = _pareto_mask(x, y)
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    scatter = ax.scatter(x, y, c=total_extension, cmap="plasma", s=75, alpha=0.82)
    ax.scatter(
        x[pareto], y[pareto], facecolors="none", edgecolors="black", s=150,
        linewidths=1.3, label="Open-chain nondominated point(s)",
    )
    for row in results.loc[pareto].itertuples(index=False):
        ax.annotate(
            f"{row.link1_mm:.0f}/{row.link2_mm:.0f}",
            (row.tracking_rmse_mm, row.combined_rms_nm),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    ax.scatter(
        exo["tracking_rmse_mm"], exo["combined_rms_nm"], marker="*", s=240,
        color="#2f6fbb", edgecolor="black", linewidth=0.8, label="EXO reference", zorder=5,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Total link extension (mm)")
    ax.set_xlabel("Ankle tracking RMSE (mm)")
    ax.set_ylabel("Combined RMS torque (Nm)")
    ax.set_title("Tracking–torque trade-off across open-chain link lengths", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "open_chain_link_sweep_pareto.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "open_chain_link_sweep_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def save_selected_profiles(
    results: pd.DataFrame,
    exo_cycle: pd.DataFrame,
    cycles_by_case: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    selected_rows = [
        ("Zero extension", results.iloc[0]),
        ("Minimum Motor 1 peak", results.loc[results["motor1_peak_nm"].idxmin()]),
        ("Maximum extension", results.iloc[-1]),
    ]
    unique: list[tuple[str, pd.Series]] = []
    used: set[str] = set()
    for label, row in selected_rows:
        case_name = str(row["case_name"])
        if case_name not in used:
            unique.append((label, row))
            used.add(case_name)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), sharex=True)
    phase_exo = np.linspace(0.0, 100.0, len(exo_cycle), endpoint=False)
    exo_error = 1000.0 * np.hypot(
        exo_cycle["ankle_actual_x"] - exo_cycle["ankle_target_x"],
        exo_cycle["ankle_actual_z"] - exo_cycle["ankle_target_z"],
    )
    axes[0].plot(phase_exo, exo_error, linewidth=2.2, color="#2f6fbb", label="EXO reference")
    axes[1].plot(phase_exo, exo_cycle["motor1_torque"], linewidth=2.2, color="#2f6fbb", label="EXO reference")
    axes[2].plot(phase_exo, exo_cycle["motor2_torque"], linewidth=2.2, color="#2f6fbb", label="EXO reference")

    colors = ["#777777", "#d4782c", "#2f8f5b"]
    for color, (selection_label, row) in zip(colors, unique):
        cycle = cycles_by_case[str(row["case_name"])]
        phase = np.linspace(0.0, 100.0, len(cycle), endpoint=False)
        error = 1000.0 * np.hypot(
            cycle["ankle_actual_x"] - cycle["ankle_target_x"],
            cycle["ankle_actual_z"] - cycle["ankle_target_z"],
        )
        label = f"{selection_label}: {row['link1_mm']:.0f}/{row['link2_mm']:.0f} mm"
        axes[0].plot(phase, error, linewidth=1.6, color=color, label=label)
        axes[1].plot(phase, cycle["motor1_torque"], linewidth=1.6, color=color, label=label)
        axes[2].plot(phase, cycle["motor2_torque"], linewidth=1.6, color=color, label=label)

    titles = ["Ankle tracking error", "Motor 1 torque", "Motor 2 torque"]
    ylabels = ["Position error (mm)", "Torque (Nm)", "Torque (Nm)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    axes[-1].set_xlabel("Gait cycle (%)")
    fig.suptitle(
        "Selected link configurations — last simulated cycle",
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.92, bottom=0.07, hspace=0.34)
    fig.savefig(output_dir / "open_chain_link_sweep_profiles.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "open_chain_link_sweep_profiles.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(
    results: pd.DataFrame,
    exo: pd.Series,
    nominal_link1_mm: float,
    nominal_link2_mm: float,
    extension_values: np.ndarray,
    cycle_duration: float,
    output_dir: Path,
) -> None:
    metrics = [
        ("tracking_rmse_mm", "Tracking RMSE (mm)"),
        ("combined_rms_nm", "Combined RMS (Nm)"),
        ("motor1_rms_nm", "Motor 1 RMS (Nm)"),
        ("motor2_rms_nm", "Motor 2 RMS (Nm)"),
        ("motor1_peak_nm", "Motor 1 peak (Nm)"),
        ("motor2_peak_nm", "Motor 2 peak (Nm)"),
    ]
    lines = [
        "# Open-chain 링크 연장 sweep",
        "",
        f"- EXO 정렬 기준 길이: Link 1 {nominal_link1_mm:.1f} mm / Link 2 {nominal_link2_mm:.1f} mm",
        f"- 각 링크 연장량: {', '.join(f'{value:.0f}' for value in extension_values)} mm",
        f"- 보행 입력 시간: {cycle_duration:.2f} s/gait cycle",
        f"- 조합 수: {len(results)}개",
        "- 영상 렌더링: 수행하지 않음",
        "",
        "## 지표별 최소값",
        "",
        "| 지표 | Link 1 / Link 2 (mm) | Open-chain 최소값 | EXO 기준값 |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in metrics:
        row = results.loc[results[metric].idxmin()]
        lines.append(
            f"| {label} | {row['link1_mm']:.1f} / {row['link2_mm']:.1f} | "
            f"{row[metric]:.3f} | {exo[metric]:.3f} |"
        )
    lines.extend(
        [
            "",
            "> 각 행의 최소값은 서로 다른 링크 조합에서 나올 수 있으므로 하나의 최적 설계를 의미하지 않는다. "
            "Tracking–torque 동시 비교는 Pareto 그래프를 사용한다.",
        ]
    )
    zero = results.iloc[0]
    maximum = results.iloc[-1]
    tracking_change = 100.0 * (maximum["tracking_rmse_mm"] / zero["tracking_rmse_mm"] - 1.0)
    rms_change = 100.0 * (maximum["combined_rms_nm"] / zero["combined_rms_nm"] - 1.0)
    motor2_peak_change = 100.0 * (maximum["motor2_peak_nm"] / zero["motor2_peak_nm"] - 1.0)
    lines.extend(
        [
            "",
            "## 범위 양 끝 비교",
            "",
            "| 지표 | 0/0 mm 연장 | 100/100 mm 연장 | 변화 |",
            "|---|---:|---:|---:|",
            f"| Tracking RMSE (mm) | {zero['tracking_rmse_mm']:.3f} | "
            f"{maximum['tracking_rmse_mm']:.3f} | +{tracking_change:.1f}% |",
            f"| Combined RMS (Nm) | {zero['combined_rms_nm']:.3f} | "
            f"{maximum['combined_rms_nm']:.3f} | +{rms_change:.1f}% |",
            f"| Motor 2 peak (Nm) | {zero['motor2_peak_nm']:.3f} | "
            f"{maximum['motor2_peak_nm']:.3f} | +{motor2_peak_change:.1f}% |",
            "",
            "현재의 위쪽 연장 범위에서는 0/0 mm 연장 조합이 tracking RMSE와 combined RMS를 "
            "동시에 최소화했다. 따라서 추가 길이가 작업공간이나 장착 간섭에 필요하지 않다면, "
            "링크 연장은 동역학 성능 측면에서 이점이 없었다.",
        ]
    )
    (output_dir / "sweep_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait-csv", type=Path, default=DEFAULT_GAIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=float, default=0.856)
    parser.add_argument("--weight", type=float, default=12.0)
    parser.add_argument("--max-extension-mm", type=float, default=100.0)
    parser.add_argument("--step-mm", type=float, default=20.0)
    parser.add_argument("--cycle-duration", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.gait_csv.is_absolute():
        args.gait_csv = (ROOT / args.gait_csv).resolve()
    else:
        args.gait_csv = args.gait_csv.resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    else:
        args.output_dir = args.output_dir.resolve()
    try:
        args.output_dir.relative_to(cfg.PATHS["output_dir"].resolve())
    except ValueError as exc:
        raise ValueError(
            "output-dir은 Git에서 제외되는 outputs/ 디렉터리 아래여야 합니다."
        ) from exc

    if args.max_extension_mm < 0 or args.step_mm <= 0:
        raise ValueError("max-extension-mm은 0 이상, step-mm은 양수여야 합니다.")
    extension_values = np.arange(0.0, args.max_extension_mm + args.step_mm * 0.5, args.step_mm)
    if not np.isclose(extension_values[-1], args.max_extension_mm):
        extension_values = np.append(extension_values, args.max_extension_mm)
    extension_values = np.unique(np.round(extension_values, 9))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nominal_link1 = 0.245 * args.height
    nominal_link2 = 0.246 * args.height

    exo_csv = _run_case(
        robot_type="exoskeleton",
        gait_csv=args.gait_csv,
        output_dir=args.output_dir,
        height=args.height,
        weight=args.weight,
        link1=nominal_link1,
        link2=nominal_link2,
        cycles=args.cycles,
        cycle_duration=args.cycle_duration,
        case_name="exo_reference",
        force=args.force,
    )
    exo_values, exo_cycle = compute_metrics(exo_csv, args.cycles)
    exo = pd.Series(exo_values)

    rows: list[dict[str, float | str]] = []
    cycles_by_case: dict[str, pd.DataFrame] = {}
    total = len(extension_values) ** 2
    index = 0
    for extension1_mm in extension_values:
        for extension2_mm in extension_values:
            index += 1
            link1 = nominal_link1 + extension1_mm / 1000.0
            link2 = nominal_link2 + extension2_mm / 1000.0
            case_name = f"open_chain_E1_{extension1_mm:05.1f}_E2_{extension2_mm:05.1f}"
            print(f"[{index:02d}/{total:02d}] L1={link1 * 1000:.1f} mm, L2={link2 * 1000:.1f} mm")
            csv_path = _run_case(
                robot_type="open_chain",
                gait_csv=args.gait_csv,
                output_dir=args.output_dir,
                height=args.height,
                weight=args.weight,
                link1=link1,
                link2=link2,
                cycles=args.cycles,
                cycle_duration=args.cycle_duration,
                case_name=case_name,
                force=args.force,
            )
            values, last_cycle = compute_metrics(csv_path, args.cycles)
            rows.append(
                {
                    "case_name": case_name,
                    "link1_extension_mm": extension1_mm,
                    "link2_extension_mm": extension2_mm,
                    "link1_mm": link1 * 1000.0,
                    "link2_mm": link2 * 1000.0,
                    **values,
                }
            )
            cycles_by_case[case_name] = last_cycle

    results = pd.DataFrame(rows).sort_values(["link1_extension_mm", "link2_extension_mm"])
    results.to_csv(args.output_dir / "open_chain_link_sweep_metrics.csv", index=False)
    pd.DataFrame([{"robot_type": "exoskeleton", **exo_values}]).to_csv(
        args.output_dir / "exo_reference_metrics.csv", index=False
    )
    save_heatmaps(
        results,
        exo,
        extension_values,
        nominal_link1 * 1000.0,
        nominal_link2 * 1000.0,
        args.output_dir,
        args.height,
        args.weight,
    )
    save_pareto_figure(results, exo, args.output_dir)
    save_selected_profiles(results, exo_cycle, cycles_by_case, args.output_dir)
    write_summary(
        results,
        exo,
        nominal_link1 * 1000.0,
        nominal_link2 * 1000.0,
        extension_values,
        args.cycle_duration,
        args.output_dir,
    )
    metadata = {
        "gait_csv": str(args.gait_csv.relative_to(ROOT)),
        "height_m": args.height,
        "weight_kg": args.weight,
        "nominal_exo_link_lengths_m": [nominal_link1, nominal_link2],
        "max_extension_mm": args.max_extension_mm,
        "step_mm": args.step_mm,
        "extension_values_mm": extension_values.tolist(),
        "cycle_duration_s": args.cycle_duration,
        "cycles": args.cycles,
        "rendered_video": False,
    }
    (args.output_dir / "sweep_input.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved sweep results: {args.output_dir}")


if __name__ == "__main__":
    main()
