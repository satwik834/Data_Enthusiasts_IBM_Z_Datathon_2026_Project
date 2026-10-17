"""
MIMIC-IV-ED Real Clinical Triage Processor
Cleans and standardizes genuine emergency department patient triage records from PhysioNet MIMIC-IV.
Calculates NEWS2 clinical risk scores on real patients and maps hospital chief complaints.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from pipeline.news2_engine import calculate_news2


def process_mimic_triage(csv_path: str) -> pd.DataFrame:
    """Processes authentic MIMIC-IV ED triage CSV, converting units and evaluating NEWS2."""
    df = pd.read_csv(csv_path)

    # Filter out rows missing vital signs
    vital_cols = ["heartrate", "resprate", "o2sat", "sbp", "temperature"]
    df_clean = df.dropna(subset=vital_cols).copy()

    # Convert Temperature from Fahrenheit to Celsius: C = (F - 32) * 5/9
    df_clean["temp_c"] = ((df_clean["temperature"].astype(float) - 32.0) * (5.0 / 9.0)).round(1)

    # Clean vitals
    df_clean["heart_rate"] = df_clean["heartrate"].astype(float).round(1)
    df_clean["respiration_rate"] = df_clean["resprate"].astype(float).round(1)
    df_clean["spo2"] = df_clean["o2sat"].astype(float).round(1)
    df_clean["systolic_bp"] = df_clean["sbp"].astype(float).round(1)
    df_clean["diastolic_bp"] = pd.to_numeric(df_clean["dbp"], errors="coerce").fillna(70.0).round(1)

    # Determine consciousness based on chief complaint / altered mental status
    def infer_consciousness(row):
        cc = str(row.get("chiefcomplaint", "")).lower()
        if any(term in cc for term in ["arrest", "unresponsive", "obtunded"]):
            return "U"
        elif any(term in cc for term in ["confusion", "altered", "seizure", "stroke", "droop"]):
            return "V"
        return "A"

    df_clean["consciousness"] = df_clean.apply(infer_consciousness, axis=1)

    # Supplemental O2 if hypoxic (SpO2 <= 92%)
    df_clean["supplemental_o2"] = df_clean["spo2"] <= 92.0

    # Calculate NEWS2 on each genuine patient
    news2_scores = []
    risk_levels = []
    icu_alerts = []

    for _, row in df_clean.iterrows():
        res = calculate_news2(
            respiration_rate=row["respiration_rate"],
            spo2=row["spo2"],
            supplemental_o2=row["supplemental_o2"],
            systolic_bp=row["systolic_bp"],
            heart_rate=row["heart_rate"],
            consciousness=row["consciousness"],
            temperature=row["temp_c"]
        )
        news2_scores.append(res["news2_score"])
        risk_levels.append(res["risk_level"])
        # High acuity (ESI 1 or 2) or NEWS2 >= 7 triggers ICU alert
        acuity_val = float(row.get("acuity", 3.0))
        icu_alerts.append(bool(res["icu_alert"] or acuity_val <= 1.0))

    df_clean["news2_score"] = news2_scores
    df_clean["risk_level"] = risk_levels
    df_clean["icu_alert"] = icu_alerts

    return df_clean


def get_real_patient_samples(csv_path: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Returns a list of structured real MIMIC-IV patients for live dashboard selection."""
    df = process_mimic_triage(csv_path)
    samples = []
    for _, row in df.head(limit).iterrows():
        samples.append({
            "subject_id": int(row["subject_id"]),
            "stay_id": int(row["stay_id"]),
            "chief_complaint": str(row.get("chiefcomplaint", "Unknown")),
            "acuity_esi": int(row.get("acuity", 3)),
            "heart_rate": float(row["heart_rate"]),
            "spo2": float(row["spo2"]),
            "systolic_bp": float(row["systolic_bp"]),
            "respiration_rate": float(row["respiration_rate"]),
            "temperature_c": float(row["temp_c"]),
            "temperature_f": float(row["temperature"]),
            "consciousness": row["consciousness"],
            "supplemental_o2": bool(row["supplemental_o2"]),
            "news2_score": int(row["news2_score"]),
            "risk_level": row["risk_level"],
            "icu_alert": bool(row["icu_alert"])
        })
    return samples


if __name__ == "__main__":
    csv_file = Path(__file__).resolve().parent.parent / "data" / "mimic_ed_triage.csv"
    processed = process_mimic_triage(str(csv_file))
    print(f"Processed {len(processed)} authentic MIMIC-IV ED records.")
    print(processed[["subject_id", "chiefcomplaint", "heart_rate", "spo2", "temp_c", "news2_score", "risk_level"]].head(10))
