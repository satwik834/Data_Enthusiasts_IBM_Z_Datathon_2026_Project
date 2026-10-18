"""
Unit Tests for Phase 2 Model & Sub-Millisecond Inference Latency
Validates model predictions, serialization, and <1ms execution timing.
"""

import time
import joblib
import unittest
import numpy as np
from pathlib import Path


class TestNEWS2Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = Path(__file__).resolve().parent.parent
        cls.model_path = base_dir / "model" / "news2_model.joblib"
        cls.assertTrue(cls, cls.model_path.exists(), "news2_model.joblib must exist")
        pkg = joblib.load(cls.model_path)
        cls.scaler = pkg["scaler"]
        cls.model = pkg["model"]

    def test_stable_prediction(self):
        # Healthy vitals: HR=72, SpO2=98, SBP=120, DBP=80, RR=16, Temp=37.0, SI=0.60, PP=40, Pain=0, Consciousness=0, O2=0, Complaint=0
        vec = np.array([[72.0, 98.0, 120.0, 80.0, 16.0, 37.0, 0.60, 40.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        vec_scaled = self.scaler.transform(vec)
        pred = self.model.predict(vec_scaled)[0]
        self.assertEqual(pred, 0, "Healthy vitals should predict Ward/Stable (0)")

    def test_sepsis_shock_prediction(self):
        # Critical sepsis: HR=140, SpO2=88, SBP=75, DBP=45, RR=32, Temp=39.8, SI=1.87, PP=30, Pain=8, Consciousness=2, O2=1, Complaint=1
        vec = np.array([[140.0, 88.0, 75.0, 45.0, 32.0, 39.8, 1.87, 30.0, 8.0, 2.0, 1.0, 1.0]], dtype=np.float32)
        vec_scaled = self.scaler.transform(vec)
        pred = self.model.predict(vec_scaled)[0]
        self.assertEqual(pred, 1, "Critical sepsis should trigger High Acuity / ICU Alert (1)")

    def test_sub_millisecond_inference_latency(self):
        # Single-sample inference latency benchmark (12 features)
        vec = np.array([[110.0, 92.0, 95.0, 60.0, 22.0, 38.2, 1.16, 35.0, 4.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        
        # Warmup
        for _ in range(50):
            vec_s = self.scaler.transform(vec)
            _ = self.model.predict(vec_s)

        # 100 iterations
        t0 = time.perf_counter_ns()
        for _ in range(100):
            vec_s = self.scaler.transform(vec)
            _ = self.model.predict(vec_s)
        t1 = time.perf_counter_ns()

        avg_latency_ms = ((t1 - t0) / 100) / 1_000_000.0
        print(f"\n[Test] Single-sample inference latency: {avg_latency_ms:.3f} ms")
        self.assertLess(avg_latency_ms, 1.0, "Inference latency must be sub-millisecond (< 1.0 ms)")


if __name__ == "__main__":
    unittest.main()
