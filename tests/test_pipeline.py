"""
Unit Tests for Real-Time Triage Telemetry Pipeline
Validates clinical NEWS2 rules, boundary conditions, fleet dispatch, and rolling window math.
"""

import unittest
from pipeline.news2_engine import (
    score_respiration_rate,
    score_spo2,
    score_oxygen,
    score_systolic_bp,
    score_heart_rate,
    score_consciousness,
    score_temperature,
    calculate_news2
)
from pipeline.dispatch_generator import AmbulanceDispatchUnit, haversine_distance_km
from pipeline.stream_manager import PatientStreamBuffer


class TestClinicalNEWS2Engine(unittest.TestCase):
    def test_respiration_rate_boundaries(self):
        self.assertEqual(score_respiration_rate(8), 3)
        self.assertEqual(score_respiration_rate(9), 1)
        self.assertEqual(score_respiration_rate(11), 1)
        self.assertEqual(score_respiration_rate(12), 0)
        self.assertEqual(score_respiration_rate(20), 0)
        self.assertEqual(score_respiration_rate(21), 2)
        self.assertEqual(score_respiration_rate(24), 2)
        self.assertEqual(score_respiration_rate(25), 3)

    def test_spo2_boundaries(self):
        self.assertEqual(score_spo2(91), 3)
        self.assertEqual(score_spo2(92), 2)
        self.assertEqual(score_spo2(93), 2)
        self.assertEqual(score_spo2(94), 1)
        self.assertEqual(score_spo2(95), 1)
        self.assertEqual(score_spo2(96), 0)
        self.assertEqual(score_spo2(100), 0)

    def test_oxygen_supplementation(self):
        self.assertEqual(score_oxygen(False), 0)
        self.assertEqual(score_oxygen(True), 2)

    def test_systolic_bp_boundaries(self):
        self.assertEqual(score_systolic_bp(90), 3)
        self.assertEqual(score_systolic_bp(91), 2)
        self.assertEqual(score_systolic_bp(100), 2)
        self.assertEqual(score_systolic_bp(101), 1)
        self.assertEqual(score_systolic_bp(110), 1)
        self.assertEqual(score_systolic_bp(111), 0)
        self.assertEqual(score_systolic_bp(219), 0)
        self.assertEqual(score_systolic_bp(220), 3)

    def test_heart_rate_boundaries(self):
        self.assertEqual(score_heart_rate(40), 3)
        self.assertEqual(score_heart_rate(41), 1)
        self.assertEqual(score_heart_rate(50), 1)
        self.assertEqual(score_heart_rate(51), 0)
        self.assertEqual(score_heart_rate(90), 0)
        self.assertEqual(score_heart_rate(91), 1)
        self.assertEqual(score_heart_rate(110), 1)
        self.assertEqual(score_heart_rate(111), 2)
        self.assertEqual(score_heart_rate(130), 2)
        self.assertEqual(score_heart_rate(131), 3)

    def test_consciousness_avpu(self):
        self.assertEqual(score_consciousness('A'), 0)
        self.assertEqual(score_consciousness('ALERT'), 0)
        self.assertEqual(score_consciousness('V'), 3)
        self.assertEqual(score_consciousness('P'), 3)
        self.assertEqual(score_consciousness('U'), 3)

    def test_temperature_boundaries(self):
        self.assertEqual(score_temperature(35.0), 3)
        self.assertEqual(score_temperature(35.1), 1)
        self.assertEqual(score_temperature(36.0), 1)
        self.assertEqual(score_temperature(36.1), 0)
        self.assertEqual(score_temperature(38.0), 0)
        self.assertEqual(score_temperature(38.1), 1)
        self.assertEqual(score_temperature(39.0), 1)
        self.assertEqual(score_temperature(39.1), 2)

    def test_aggregate_news2_low_risk(self):
        # Perfectly healthy physiological parameters
        res = calculate_news2(
            respiration_rate=16,
            spo2=98,
            supplemental_o2=False,
            systolic_bp=120,
            heart_rate=72,
            consciousness='A',
            temperature=37.0
        )
        self.assertEqual(res["news2_score"], 0)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertFalse(res["icu_alert"])
        self.assertFalse(res["has_red_score"])

    def test_aggregate_news2_high_risk_icu_alert(self):
        # Severe septic shock
        res = calculate_news2(
            respiration_rate=28,   # 3
            spo2=89,               # 3
            supplemental_o2=True,  # 2
            systolic_bp=80,        # 3
            heart_rate=135,        # 3
            consciousness='V',     # 3
            temperature=39.5       # 2
        )
        self.assertEqual(res["news2_score"], 19)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertTrue(res["icu_alert"])

    def test_red_score_triggers_medium_risk(self):
        # Only RR is 25 (score 3), rest are 0 -> Total score 3, but single parameter 3 triggers MEDIUM risk
        res = calculate_news2(
            respiration_rate=25,   # 3
            spo2=98,               # 0
            supplemental_o2=False, # 0
            systolic_bp=120,       # 0
            heart_rate=70,         # 0
            consciousness='A',     # 0
            temperature=37.0       # 0
        )
        self.assertEqual(res["news2_score"], 3)
        self.assertTrue(res["has_red_score"])
        self.assertEqual(res["risk_level"], "MEDIUM")


class TestDispatchSimulator(unittest.TestCase):
    def test_distance_and_transit(self):
        unit = AmbulanceDispatchUnit("AMB-101", start_lat=40.7128, start_lon=-74.0060)
        self.assertGreater(unit.total_distance_km, 0.5)

        initial_dist = unit.total_distance_km
        state = unit.step(dt_seconds=10.0, high_priority=True)
        self.assertLess(state["remaining_distance_km"], initial_dist)
        self.assertTrue(state["siren_active"])
        self.assertGreater(state["eta_minutes"], 0.0)


class TestStreamBuffer(unittest.TestCase):
    def test_rolling_window_and_slopes(self):
        buf = PatientStreamBuffer("PATIENT-001", max_window_seconds=30.0)
        import time
        now = time.time()

        r1 = {"timestamp": now, "heart_rate": 80, "spo2": 98, "systolic_bp": 120, "respiration_rate": 16, "news2_score": 0}
        r2 = {"timestamp": now + 5, "heart_rate": 95, "spo2": 94, "systolic_bp": 115, "respiration_rate": 20, "news2_score": 2}
        r3 = {"timestamp": now + 10, "heart_rate": 120, "spo2": 89, "systolic_bp": 95, "respiration_rate": 26, "news2_score": 8}

        buf.add_reading(r1)
        buf.add_reading(r2)
        features = buf.add_reading(r3)

        self.assertEqual(features["window_size"], 3)
        self.assertGreater(features["hr_slope_per_sec"], 0.0)
        self.assertLess(features["spo2_slope_per_sec"], 0.0)
        self.assertTrue(features["acute_decompensation_warning"])


if __name__ == "__main__":
    unittest.main()
