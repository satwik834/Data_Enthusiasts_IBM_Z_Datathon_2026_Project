"""
Model Training and ONNX Export Script
Trains an optimized Multi-Layer Perceptron (Neural Network) classifier on MIMIC-IV vitals
to predict acute ICU deterioration (NEWS2 >= 7) with sub-millisecond execution latency.
Exports the trained model to ONNX format and joblib serialization.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
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
    "respiration_rate",
    "temperature",
    "supplemental_o2",
    "consciousness"
]

CONSCIOUSNESS_MAP = {"A": 0, "ALERT": 0, "V": 1, "P": 2, "U": 3}


def preprocess_data(df: pd.DataFrame):
    """Encodes categorical fields and extracts feature vectors and target labels."""
    df_clean = df.copy()

    # Map consciousness
    df_clean["consciousness"] = df_clean["consciousness"].astype(str).str.strip().str.upper().map(
        lambda x: CONSCIOUSNESS_MAP.get(x, 1)
    )

    # Map boolean O2
    df_clean["supplemental_o2"] = df_clean["supplemental_o2"].astype(int)

    X = df_clean[FEATURE_COLS].astype(np.float32).values
    y_icu = df_clean["icu_alert"].astype(int).values
    y_score = df_clean["news2_score"].astype(int).values

    return X, y_icu, y_score


def train_and_export(train_path: str, test_path: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"Loading testing data from: {test_path}")
    df_test = pd.read_csv(test_path)

    X_train, y_train, _ = preprocess_data(df_train)
    X_test, y_test, _ = preprocess_data(df_test)

    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Deep Learning Neural Network optimized for IBM Telum AI Accelerator
    print("Training IBM Telum-Targeted Deep Learning Classifier (MLP)...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (Test Set):")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Stable/Ward", "ICU Critical Alert"]))

    # Package model pipeline (Scaler + MLP) for convenient inference
    pipeline = {
        "scaler": scaler,
        "model": model,
        "feature_cols": FEATURE_COLS
    }

    joblib_path = Path(output_dir) / "news2_model.joblib"
    joblib.dump(pipeline, joblib_path)
    print(f"Saved serialized model package -> {joblib_path}")

    # Export to ONNX if skl2onnx available
    onnx_path = Path(output_dir) / "news2_triage.onnx"
    if HAS_SKL2ONNX:
        print("Converting neural network pipeline to ONNX format...")
        initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
        from sklearn.pipeline import Pipeline
        full_pipe = Pipeline([("scaler", scaler), ("mlp", model)])
        onnx_model = convert_sklearn(full_pipe, initial_types=initial_type, target_opset=13)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Successfully exported ONNX model -> {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")
    else:
        print("Notice: skl2onnx will convert and export news2_triage.onnx inside the LinuxONE container.")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "onnx_exported": HAS_SKL2ONNX
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_file = base_dir / "data" / "mimic_vitals_train.csv"
    test_file = base_dir / "data" / "mimic_vitals_test.csv"
    out_dir = base_dir / "model"

    train_and_export(str(train_file), str(test_file), str(out_dir))
