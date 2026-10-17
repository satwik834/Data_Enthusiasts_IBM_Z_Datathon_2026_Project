"""
Dataset Generation Script
Synthesizes 10,000 training and 2,500 test samples of clinical MIMIC-IV vitals with NEWS2 labels.
Exports CSV files locally to data/ and syncs to /home/linux1/shared/ on the LinuxONE VM.
"""

import os
import csv
import random
from pathlib import Path
from pipeline.vitals_generator import generate_single_sample


def generate_dataset(num_samples: int, output_filepath: str):
    """Generates a balanced dataset of clinical biometric records."""
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "heart_rate",
        "spo2",
        "systolic_bp",
        "respiration_rate",
        "temperature",
        "supplemental_o2",
        "consciousness",
        "clinical_profile",
        "news2_score",
        "risk_level",
        "icu_alert",
        "has_red_score"
    ]

    profiles = ["STABLE", "DETERIORATING", "SEPSIS_SHOCK", "RESPIRATORY_FAILURE"]

    with open(output_filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(num_samples):
            # Equal sampling across clinical conditions for balanced learning
            p = random.choice(profiles)
            row = generate_single_sample(p)
            # Remove transient keys
            row.pop("timestamp", None)
            row.pop("patient_id", None)
            row.pop("step", None)
            writer.writerow(row)

    print(f"Successfully generated {num_samples} samples -> {output_filepath}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "data" / "mimic_vitals_train.csv"
    test_path = base_dir / "data" / "mimic_vitals_test.csv"

    print("Generating training dataset (10,000 samples)...")
    generate_dataset(10000, str(train_path))

    print("Generating testing dataset (2,500 samples)...")
    generate_dataset(2500, str(test_path))

    print("Dataset generation complete!")
