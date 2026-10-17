"""
In-Memory Rolling Window Stream Preprocessor
Maintains a 30-second sliding window per patient to compute moving averages and acute deterioration slopes.
"""

from collections import deque
import numpy as np
from typing import Dict, Any, List, Optional


class PatientStreamBuffer:
    def __init__(self, patient_id: str, max_window_seconds: float = 30.0):
        self.patient_id = patient_id
        self.max_window_seconds = max_window_seconds
        self.buffer = deque()

    def add_reading(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """Adds a new reading and evicts entries older than the window duration."""
        current_ts = reading.get("timestamp", 0.0)
        self.buffer.append(reading)

        # Evict stale entries beyond window
        cutoff = current_ts - self.max_window_seconds
        while self.buffer and self.buffer[0].get("timestamp", 0.0) < cutoff:
            self.buffer.popleft()

        return self.compute_window_features()

    def compute_window_features(self) -> Dict[str, Any]:
        """Computes statistical rolling aggregations and trend velocities."""
        if not self.buffer:
            return {}

        hrs = [r["heart_rate"] for r in self.buffer]
        spo2s = [r["spo2"] for r in self.buffer]
        sbps = [r["systolic_bp"] for r in self.buffer]
        rrs = [r["respiration_rate"] for r in self.buffer]
        news2_scores = [r["news2_score"] for r in self.buffer]

        latest = self.buffer[-1]
        n = len(self.buffer)

        # Deterioration trend (slope between oldest and newest in buffer)
        if n >= 2:
            dt = max(1.0, self.buffer[-1]["timestamp"] - self.buffer[0]["timestamp"])
            hr_slope = (hrs[-1] - hrs[0]) / dt
            spo2_slope = (spo2s[-1] - spo2s[0]) / dt
            sbp_slope = (sbps[-1] - sbps[0]) / dt
            rr_slope = (rrs[-1] - rrs[0]) / dt
        else:
            hr_slope = spo2_slope = sbp_slope = rr_slope = 0.0

        # Rapid deterioration flag: falling oxygen with accelerating heart rate
        acute_warning = bool(spo2_slope < -0.05 and hr_slope > 0.10)

        return {
            "patient_id": self.patient_id,
            "window_size": n,
            "latest_vitals": latest,
            "hr_mean_30s": round(float(np.mean(hrs)), 1),
            "hr_std_30s": round(float(np.std(hrs)), 1),
            "hr_slope_per_sec": round(hr_slope, 3),
            "spo2_mean_30s": round(float(np.mean(spo2s)), 1),
            "spo2_slope_per_sec": round(spo2_slope, 3),
            "sbp_mean_30s": round(float(np.mean(sbps)), 1),
            "sbp_slope_per_sec": round(sbp_slope, 3),
            "rr_mean_30s": round(float(np.mean(rrs)), 1),
            "rr_slope_per_sec": round(rr_slope, 3),
            "news2_max_30s": int(np.max(news2_scores)),
            "acute_decompensation_warning": acute_warning
        }
