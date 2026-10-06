"""Headless regression tests for the three rehabilitation modes."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config as cfg
import simulate


class PulseEffortControl(dict):
    """힘 기준을 잠깐만 넘겨 swing gate의 latch 동작을 시험한다."""

    effort_reads = 0

    def get(self, key, default=None):
        if key == "patient_effort":
            self.effort_reads += 1
            return 40.0 if 30 <= self.effort_reads <= 32 else 0.0
        return super().get(key, default)


class RehabilitationModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_xml_path = cfg.PATHS["xml_model"]
        self.old_output_dir = cfg.PATHS["output_dir"]
        self.old_plot_all = simulate.plot_all
        temp_path = Path(self.temp_dir.name)
        cfg.PATHS["xml_model"] = temp_path / "mode_test.xml"
        cfg.PATHS["output_dir"] = temp_path / "outputs"
        simulate.plot_all = None

    def tearDown(self) -> None:
        cfg.PATHS["xml_model"] = self.old_xml_path
        cfg.PATHS["output_dir"] = self.old_output_dir
        simulate.plot_all = self.old_plot_all
        self.temp_dir.cleanup()

    @staticmethod
    def _control(mode: str, *, effort: float = 0.0) -> dict:
        return {
            "requested_state": mode,
            "stop_requested": False,
            "cadence_spm": 40.0,
            "soft_start_duration": 0.5,
            "patient_effort": effort,
            "initiation_threshold": 20.0,
            "sit_to_stand_duration": 1.5,
            "sit_to_stand_request_id": 1 if mode == "sit_to_stand" else 0,
            "right_hip_support": 100.0,
            "right_knee_support": 100.0,
            "left_hip_support": 100.0,
            "left_knee_support": 100.0,
            "alarm": "",
        }

    def _run(
        self,
        mode: str,
        cycles: int,
        *,
        effort: float = 0.0,
        control: dict | None = None,
    ) -> dict:
        control = control or self._control(mode, effort=effort)
        simulate.run_simulation(
            render_mode="none",
            n_cycles=cycles,
            robot_type="exoskeleton",
            out_dir_name=mode,
            save_video=False,
            runtime_control=control,
            right_hip_range_deg=(-15.0, 50.0),
            right_knee_range_deg=(0.0, 80.0),
            left_hip_range_deg=(-15.0, 50.0),
            left_knee_range_deg=(0.0, 80.0),
        )
        self.assertEqual(control.get("alarm"), "")
        return control

    def test_automatic_advances_reference_gait(self) -> None:
        control = self._run("automatic", 1)
        self.assertGreater(float(control["gait_phase_pct"]), 10.0)
        self.assertFalse(bool(control["active_gate_blocked"]))

    def test_active_waits_at_swing_threshold_without_effort(self) -> None:
        control = self._run("active", 1, effort=0.0)
        self.assertAlmostEqual(float(control["gait_phase_pct"]), 10.0, delta=0.2)
        self.assertTrue(bool(control["active_gate_blocked"]))

    def test_active_effort_pulse_releases_only_current_swing(self) -> None:
        control = PulseEffortControl(self._control("active"))
        self._run("active", 4, control=control)
        self.assertAlmostEqual(float(control["gait_phase_pct"]), 60.0, delta=0.2)
        self.assertTrue(bool(control["active_gate_blocked"]))

    def test_sit_to_stand_completes_one_repetition(self) -> None:
        control = self._run("sit_to_stand", 4)
        self.assertEqual(control["sit_to_stand_stage"], "complete")
        self.assertEqual(int(control["sit_to_stand_repetitions"]), 1)
        self.assertAlmostEqual(float(control["gait_phase_pct"]), 100.0)


if __name__ == "__main__":
    unittest.main()
