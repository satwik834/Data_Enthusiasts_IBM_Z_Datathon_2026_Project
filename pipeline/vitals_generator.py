"""
MIMIC-IV Calibrated Streaming Vitals Generator
Synthesizes real-time biometric trajectories across clinical patient states.
"""

import time
import random
import numpy as np
from typing import Dict, Any, Generator
from pipeline.news2_engine import calculate_news2


CLINICAL_PROFILES = {
    "STABLE": {
        "hr": (60, 85),
        "spo2": (96, 99),
        "sbp": (115, 135),
        "rr": (13, 18),
        "temp": (36.5, 37.2),
        "avpu_probs": [0.99, 0.01, 0.0, 0.0],
        "o2_prob": 0.05
    },
    "DETERIORATING": {
        "hr": (95, 115),
        "spo2": (92, 95),
        "sbp": (95, 110),
        "rr": (21, 25),
        "temp": (37.8, 38.6),
        "avpu_probs": [0.85, 0.12, 0.03, 0.0],
        "o2_prob": 0.40
    },
    "SEPSIS_SHOCK": {
        "hr": (125, 145),
        "spo2": (88, 93),
        "sbp": (68, 88),
        "rr": (26, 34),
        "temp": (39.2, 40.2),
        "avpu_probs": [0.15, 0.50, 0.30, 0.05],
        "o2_prob": 0.90
    },
    "RESPIRATORY_FAILURE": {
        "hr": (110, 135),
        "spo2": (78, 89),
        "sbp": (135, 165),
        "rr": (28, 38),
        "temp": (36.7, 37.6),
        "avpu_probs": [0.20, 0.60, 0.15, 0.05],
        "o2_prob": 0.95
    }
}

AVPU_STATES = ['A', 'V', 'P', 'U']


def generate_single_sample(profile_name: str = None) -> Dict[str, Any]:
    """Generates a single biometric observation calibrated to the chosen clinical archetype."""
    if not profile_name or profile_name not in CLINICAL_PROFILES:
        profile_name = random.choices(
            ["STABLE", "DETERIORATING", "SEPSIS_SHOCK", "RESPIRATORY_FAILURE"],
            weights=[0.55, 0.20, 0.15, 0.10]
        )[0]

    prof = CLINICAL_PROFILES[profile_name]

    hr = float(np.random.normal(np.mean(prof["hr"]), (prof["hr"][1] - prof["hr"][0]) / 4.0))
    spo2 = float(np.clip(np.random.normal(np.mean(prof["spo2"]), 1.5), 65, 100))
    sbp = float(np.random.normal(np.mean(prof["sbp"]), (prof["sbp"][1] - prof["sbp"][0]) / 4.0))
    rr = float(np.clip(np.random.normal(np.mean(prof["rr"]), 2.0), 6, 50))
    temp = float(np.random.normal(np.mean(prof["temp"]), 0.3))
    o2 = bool(random.random() < prof["o2_prob"])
    avpu = str(random.choices(AVPU_STATES, weights=prof["avpu_probs"])[0])

    news2_eval = calculate_news2(
        respiration_rate=rr,
        spo2=spo2,
        supplemental_o2=o2,
        systolic_bp=sbp,
        heart_rate=hr,
        consciousness=avpu,
        temperature=temp
    )

    return {
        "timestamp": time.time(),
        "clinical_profile": profile_name,
        "heart_rate": round(hr, 1),
        "spo2": round(spo2, 1),
        "systolic_bp": round(sbp, 1),
        "respiration_rate": round(rr, 1),
        "temperature": round(temp, 1),
        "supplemental_o2": o2,
        "consciousness": avpu,
        "news2_score": news2_eval["news2_score"],
        "risk_level": news2_eval["risk_level"],
        "icu_alert": news2_eval["icu_alert"],
        "has_red_score": news2_eval["has_red_score"]
    }


def stream_patient_vitals(
    patient_id: str,
    profile_name: str = "STABLE",
    step_delay_sec: float = 1.0,
    max_steps: int = 100
) -> Generator[Dict[str, Any], None, None]:
    """Generates continuous streaming vitals with slight physiological jitter."""
    for step in range(max_steps):
        sample = generate_single_sample(profile_name)
        sample["patient_id"] = patient_id
        sample["step"] = step
        yield sample
        if step_delay_sec > 0:
            time.sleep(step_delay_sec)
