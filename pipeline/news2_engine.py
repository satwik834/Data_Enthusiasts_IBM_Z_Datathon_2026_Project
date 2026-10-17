"""
NEWS2 (National Early Warning Score 2) Clinical Engine
Standardized to Royal College of Physicians (UK) guidelines.
Provides physiological score breakdown, risk level, and emergency ICU escalation trigger.
"""

from typing import Dict, Any, Tuple


def score_respiration_rate(rr: float) -> int:
    if rr <= 8:
        return 3
    elif 9 <= rr <= 11:
        return 1
    elif 12 <= rr <= 20:
        return 0
    elif 21 <= rr <= 24:
        return 2
    else:  # >= 25
        return 3


def score_spo2(spo2: float) -> int:
    if spo2 <= 91:
        return 3
    elif 92 <= spo2 <= 93:
        return 2
    elif 94 <= spo2 <= 95:
        return 1
    else:  # >= 96
        return 0


def score_oxygen(supplemental_o2: bool) -> int:
    return 2 if supplemental_o2 else 0


def score_systolic_bp(sbp: float) -> int:
    if sbp <= 90:
        return 3
    elif 91 <= sbp <= 100:
        return 2
    elif 101 <= sbp <= 110:
        return 1
    elif 111 <= sbp <= 219:
        return 0
    else:  # >= 220
        return 3


def score_heart_rate(hr: float) -> int:
    if hr <= 40:
        return 3
    elif 41 <= hr <= 50:
        return 1
    elif 51 <= hr <= 90:
        return 0
    elif 91 <= hr <= 110:
        return 1
    elif 111 <= hr <= 130:
        return 2
    else:  # >= 131
        return 3


def score_consciousness(avpu: str) -> int:
    """
    AVPU scale:
    'A' -> Alert (0)
    'V' -> Responds to Voice (3)
    'P' -> Responds to Pain (3)
    'U' -> Unresponsive (3)
    """
    val = str(avpu).strip().upper()
    if val in ['A', 'ALERT', '0']:
        return 0
    return 3


def score_temperature(temp_c: float) -> int:
    if temp_c <= 35.0:
        return 3
    elif 35.1 <= temp_c <= 36.0:
        return 1
    elif 36.1 <= temp_c <= 38.0:
        return 0
    elif 38.1 <= temp_c <= 39.0:
        return 1
    else:  # >= 39.1
        return 2


def calculate_news2(
    respiration_rate: float,
    spo2: float,
    supplemental_o2: bool,
    systolic_bp: float,
    heart_rate: float,
    consciousness: str,
    temperature: float
) -> Dict[str, Any]:
    """
    Computes total NEWS2 score, individual sub-scores, clinical risk stratification,
    and ICU emergency alert flag according to Royal College of Physicians guidelines.
    """
    s_rr = score_respiration_rate(respiration_rate)
    s_spo2 = score_spo2(spo2)
    s_o2 = score_oxygen(supplemental_o2)
    s_sbp = score_systolic_bp(systolic_bp)
    s_hr = score_heart_rate(heart_rate)
    s_avpu = score_consciousness(consciousness)
    s_temp = score_temperature(temperature)

    subscores = {
        "respiration_rate": s_rr,
        "spo2": s_spo2,
        "oxygen": s_o2,
        "systolic_bp": s_sbp,
        "heart_rate": s_hr,
        "consciousness": s_avpu,
        "temperature": s_temp
    }

    total_score = sum(subscores.values())
    has_red_score = any(score == 3 for score in subscores.values())

    if total_score >= 7:
        risk_level = "HIGH"
        clinical_action = "Immediate emergency assessment by critical care / ICU team."
        icu_alert = True
    elif total_score >= 5 or has_red_score:
        risk_level = "MEDIUM"
        clinical_action = "Urgent review by clinician with acute care competence."
        icu_alert = False
    else:
        risk_level = "LOW"
        clinical_action = "Routine ward/transit monitoring (minimum 4-6 hourly)."
        icu_alert = False

    return {
        "news2_score": total_score,
        "risk_level": risk_level,
        "has_red_score": has_red_score,
        "icu_alert": icu_alert,
        "clinical_action": clinical_action,
        "subscores": subscores
    }
