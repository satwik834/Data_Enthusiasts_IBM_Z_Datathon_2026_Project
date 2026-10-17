"""
Unit Tests for FastAPI Inference Server
Validates REST endpoints, latency response, and clinical alert logic.
"""

import unittest
from fastapi.testclient import TestClient
from server.app import app, load_model


class TestFastAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_model()
        cls.client = TestClient(app)

    def test_health_check(self):
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("platform", data)

    def test_triage_stable(self):
        payload = {
            "heart_rate": 70.0,
            "spo2": 98.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "respiration_rate": 15.0,
            "temperature": 36.8,
            "supplemental_o2": False,
            "consciousness": "A",
            "pain_score": 1.0,
            "chief_complaint": "Routine wellness check"
        }
        res = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["risk_level"], "LOW")
        self.assertFalse(data["icu_alert"])
        self.assertFalse(data["icu_bed_reserved"])
        self.assertLess(data["inference_latency_ms"], 5.0)
        self.assertIn("ai_acuity_probability", data)
        self.assertIn("shock_index", data)
        self.assertEqual(data["shock_risk"], "NORMAL")

    def test_triage_critical(self):
        payload = {
            "heart_rate": 140.0,
            "spo2": 87.0,
            "systolic_bp": 78.0,
            "diastolic_bp": 45.0,
            "respiration_rate": 32.0,
            "temperature": 39.8,
            "supplemental_o2": True,
            "consciousness": "V",
            "pain_score": 0.0,
            "chief_complaint": "Severe dyspnea, altered mental status, septic shock"
        }
        res = self.client.post("/api/v1/triage", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["risk_level"], "HIGH")
        self.assertTrue(data["icu_alert"])
        self.assertTrue(data["icu_bed_reserved"])
        self.assertGreaterEqual(data["news2_score"], 7)
        self.assertIn("ai_acuity_probability", data)
        self.assertEqual(data["shock_risk"], "ELEVATED")

    def test_telemetry_sample(self):
        res = self.client.get("/api/v1/telemetry/sample")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("vitals", data)
        self.assertIn("dispatch", data)
        self.assertIn("lat", data["dispatch"])
        self.assertIn("lon", data["dispatch"])

    def test_benchmark_endpoint(self):
        res = self.client.get("/api/v1/benchmark")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("ibm_z_mean_ms", data)
        self.assertTrue(data["sub_millisecond_pass"])


if __name__ == "__main__":
    unittest.main()
