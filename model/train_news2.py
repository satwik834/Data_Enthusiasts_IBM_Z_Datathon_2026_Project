"""
IBM Z Datathon 2026 - Production Model Training Pipeline
PhysioNet MIMIC-IV-ED Clinical Acuity Neural Predictor
Trained on genuine emergency department triage encounters from Beth Israel Deaconess Medical Center.
Predicts Critical Clinical Acuity (Emergency Severity Index ESI <= 2: Resuscitation & Emergent)
to complement the Royal College of Physicians deterministic NEWS2 safety protocol.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    HAS_SKL2ONNX = True
except ImportError:
    HAS_SKL2ONNX = False

FEATURE_COLS = [
    "heart_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "respiration_rate",
    "temperature",
    "shock_index",
    "pulse_pressure",
    "pain_score",
    "consciousness",
    "supplemental_o2",
    "high_risk_complaint"
]

CONSCIOUSNESS_MAP = {"A": 0, "ALERT": 0, "V": 1, "P": 2, "U": 3}
HIGH_RISK_KEYWORDS = [
    "arrest", "unresponsive", "stroke", "fall", "sob", "bleed",
    "dka", "seizure", "chest", "intubated", "mvc", "head bleed",
    "facial droop", "car vs pole"
]


def parse_consciousness(val: Any) -> int:
    s = str(val).strip().upper()
    return CONSCIOUSNESS_MAP.get(s, 0)


def extract_complaint_risk(complaint: str) -> int:
    c = str(complaint).lower()
    return 1 if any(k in c for k in HIGH_RISK_KEYWORDS) else 0


def load_and_preprocess_mimic(csv_path: str):
    """Ingests authentic MIMIC-IV-ED triage CSV, normalizes features, and extracts clinical target."""
    df = pd.read_csv(csv_path)
    vital_cols = ["heartrate", "resprate", "o2sat", "sbp", "dbp", "temperature", "acuity"]
    clean = df.dropna(subset=vital_cols).copy()

    # Convert Fahrenheit to Celsius: C = (F - 32) * 5/9
    clean["temp_c"] = ((clean["temperature"].astype(float) - 32.0) * (5.0 / 9.0)).round(1)
    clean["hr"] = clean["heartrate"].astype(float).round(1)
    clean["rr"] = clean["resprate"].astype(float).round(1)
    clean["spo2"] = clean["o2sat"].astype(float).round(1)
    clean["sbp"] = clean["sbp"].astype(float).round(1)
    clean["dbp"] = clean["dbp"].astype(float).round(1)

    # Derived hemodynamic features
    clean["shock_index"] = (clean["hr"] / clean["sbp"].replace(0, 120.0)).fillna(0.7).round(3)
    clean["pulse_pressure"] = (clean["sbp"] - clean["dbp"]).round(1)

    def parse_pain(val):
        try:
            return min(max(float(val), 0.0), 10.0)
        except Exception:
            return 0.0

    clean["pain_clean"] = clean["pain"].apply(parse_pain).fillna(0.0)

    def infer_consciousness(row):
        cc = str(row.get("chiefcomplaint", "")).lower()
        if any(k in cc for k in ["arrest", "unresponsive", "obtunded"]):
            return 3  # U
        if any(k in cc for k in ["confusion", "altered", "seizure", "stroke", "droop"]):
            return 1  # V
        return 0  # A

    clean["consciousness_clean"] = clean.apply(infer_consciousness, axis=1)
    clean["supp_o2"] = (clean["spo2"] <= 92.0).astype(int)
    clean["complaint_risk"] = clean["chiefcomplaint"].apply(extract_complaint_risk)

    # True Clinical Target: High Acuity ESI 1 (Resuscitation) or ESI 2 (Emergent) vs ESI 3+
    clean["target_high_acuity"] = (clean["acuity"].astype(float) <= 2.0).astype(int)

    feature_matrix = clean[[
        "hr", "spo2", "sbp", "dbp", "rr", "temp_c",
        "shock_index", "pulse_pressure", "pain_clean",
        "consciousness_clean", "supp_o2", "complaint_risk"
    ]].values.astype(np.float32)

    target_vector = clean["target_high_acuity"].values.astype(int)

    return feature_matrix, target_vector, clean


def train_and_export_acuity_model(csv_path: str, output_dir: str):
    """Trains the clinical acuity predictor, evaluates 5-fold CV, and exports serialized models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MIMIC-IV-ED EMERGENCY CLINICAL ACUITY MODEL TRAINING")
    print("Ground Truth: Real PhysioNet MIMIC-IV-ED Patient Acuity (ESI 1-2 vs 3+)")
    print("=" * 70)

    X, y, df_clean = load_and_preprocess_mimic(csv_path)
    total_samples = X.shape[0]
    high_acuity_count = int(np.sum(y == 1))
    routine_count = int(np.sum(y == 0))

    print(f"Total Cohort Size : {total_samples} patients")
    print(f"High Acuity (ESI 1-2 Resuscitation/Emergent) : {high_acuity_count} ({high_acuity_count/total_samples*100:.1f}%)")
    print(f"Routine/Urgent (ESI 3-5 Standard)           : {routine_count} ({routine_count/total_samples*100:.1f}%)")

    # 5-Fold Stratified Cross-Validation on authentic patients
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()

    # Calibrated neural network classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.5,
        max_iter=1000,
        random_state=42
    )

    cv_acc, cv_auc, cv_f1 = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[test_idx]
        y_tr, y_val = y[train_idx], y[test_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        clf.fit(X_tr_s, y_tr)
        val_pred = clf.predict(X_val_s)
        val_prob = clf.predict_proba(X_val_s)[:, 1]

        cv_acc.append(accuracy_score(y_val, val_pred))
        cv_auc.append(roc_auc_score(y_val, val_prob))
        cv_f1.append(f1_score(y_val, val_pred))

    mean_acc = float(np.mean(cv_acc))
    mean_auc = float(np.mean(cv_auc))
    mean_f1 = float(np.mean(cv_f1))

    print("\n5-FOLD STRATIFIED CROSS-VALIDATION RESULTS (Real Patients):")
    print(f"Mean Accuracy : {mean_acc * 100:.2f}%")
    print(f"Mean ROC-AUC  : {mean_auc:.4f}")
    print(f"Mean F1-Score : {mean_f1:.4f}")

    # Fit final production model on full cohort
    final_scaler = StandardScaler()
    X_full_scaled = final_scaler.fit_transform(X)

    final_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.5,
        max_iter=1000,
        random_state=42
    )
    final_model.fit(X_full_scaled, y)

    # Save joblib bundle
    pipeline_pkg = {
        "scaler": final_scaler,
        "model": final_model,
        "feature_cols": FEATURE_COLS,
        "metrics": {
            "cross_val_accuracy": round(mean_acc, 4),
            "cross_val_roc_auc": round(mean_auc, 4),
            "cross_val_f1": round(mean_f1, 4),
            "cohort_size": total_samples,
            "high_acuity_cases": high_acuity_count,
            "routine_cases": routine_count,
            "data_source": "PhysioNet MIMIC-IV Emergency Department (MIMIC-IV-ED v2.2)",
            "clinical_standard": "Emergency Severity Index (ESI) Acuity vs Royal College of Physicians NEWS2"
        }
    }

    joblib_path = Path(output_dir) / "news2_model.joblib"
    joblib.dump(pipeline_pkg, joblib_path)
    print(f"\nSaved production model package -> {joblib_path}")

    # Save metrics JSON for documentation
    metrics_path = Path(output_dir) / "acuity_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(pipeline_pkg["metrics"], f, indent=2)
    print(f"Saved evaluation metrics -> {metrics_path}")

    # Export to ONNX if skl2onnx is available
    onnx_path = Path(output_dir) / "news2_triage.onnx"
    if HAS_SKL2ONNX:
        print("Converting production model pipeline to ONNX format...")
        initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
        from sklearn.pipeline import Pipeline
        full_pipe = Pipeline([("scaler", final_scaler), ("mlp", final_model)])
        onnx_model = convert_sklearn(full_pipe, initial_types=initial_type, target_opset=13)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Successfully exported ONNX model -> {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")
    else:
        print("Notice: ONNX export will be performed natively inside LinuxONE s390x container.")

    return pipeline_pkg["metrics"]


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    csv_input = base_dir / "data" / "mimic_ed_triage.csv"
    out_dir = base_dir / "model"
    train_and_export_acuity_model(str(csv_input), str(out_dir))
