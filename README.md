# [Data Enthusiasts_IBM Z Datathon 2026 Project]
## Sub-Millisecond IBM Z Real-Time AI Triage & Emergency Response System
### IBM Z Datathon 2026 — Track: Real-Time AI for Critical Decisions

[![Platform](https://img.shields.io/badge/Platform-IBM%20LinuxONE%20s390x-blue.svg)](https://linuxone.cloud.marist.edu)
[![Architecture](https://img.shields.io/badge/Architecture-IBM%20z16%20%7C%20Telum%20AI%20Accelerator-darkblue.svg)](https://www.ibm.com/products/z16)
[![Latency](https://img.shields.io/badge/Inference%20Latency-%3C1.0ms%20(Sub--millisecond)-brightgreen.svg)]()
[![Clinical%20Standard](https://img.shields.io/badge/Clinical%20Standard-NEWS2%20(Royal%20College%20of%20Physicians)-red.svg)]()

---

## 🚑 Executive Summary
During acute medical emergencies (cardiac arrest, respiratory failure, sepsis, severe trauma), patient transit windows represent critical moments where minutes dictate survival. Current telemedicine and cloud AI systems suffer from **high network latency (100–500ms+ jitter)** and **strict privacy compliance barriers (HIPAA/GDPR)** when streaming sensitive patient biometric telemetry across public networks.

This project delivers an enterprise-grade, end-to-end real-time emergency triage decision-support system running natively on **IBM LinuxONE (s390x)**:
* **Microsecond Inference**: Executes quantized **NEWS2 (National Early Warning Score 2)** early deterioration models directly on the IBM Telum processor architecture with **sub-millisecond (<1ms)** latency.
* **Proactive ICU Allocation**: Automatically alerts receiving hospital ICU teams and reserves beds before ambulance arrival when severe deterioration is detected.
* **Zero-Trust Biometric Security**: Keeps patient vital streams protected within the enterprise mainframe perimeter.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph S1["Field Units & Telemetry"]
        A1["Ambulance Unit Telemetry\n(GPS, Traffic, ETA)"]
        A2["Patient Biometrics\n(MIMIC-IV HR, SpO2, BP, RR, Temp, AVPU)"]
    end

    subgraph S2["IBM LinuxONE Community Cloud (s390x Architecture)"]
        B1["Ingestion & Normalization Engine\n(Rolling 30s Window)"]
        B2["ONNX Model Runtime\n(Quantized NEWS2 Scoring)"]
        B3["IBM Telum AI Acceleration\n(Microsecond Inference < 1ms)"]
        B4["FastAPI Triage Service\n(Alerts & Benchmarking)"]
        
        B1 --> B2
        B2 <--> B3
        B2 --> B4
    end

    subgraph S3["Real-Time Command Dashboard"]
        C1["Patient Vital Signs Stream\n(ECG / Vitals / Deterioration Alert)"]
        C2["Interactive Transit Map\n(Ambulance GPS & Hospital Route)"]
        C3["Hospital ICU Triage Queue\n(Bed Reservation & Team Dispatch)"]
        C4["Latency Comparison Monitor\n(LinuxONE ~0.8ms vs Cloud ~185ms)"]
    end

    A1 -->|HTTP Stream| B1
    A2 -->|HTTP Stream| B1
    B4 -->|WebSocket / SSE| C1
    B4 -->|WebSocket / SSE| C2
    B4 -->|WebSocket / SSE| C3
    B4 -->|WebSocket / SSE| C4
```

---

## ⚡ Key Highlights & Benchmark Comparison
* **Sub-Millisecond Inference**: Model execution benchmarked on IBM LinuxONE s390x vs standard cloud API round-trip:
  * **IBM LinuxONE (On-System ONNX)**: **~0.8 ms**
  * **Public Cloud REST API**: **~185 ms** (230x latency reduction)
* **Clinical Standards Compliance**: Full implementation of the Royal College of Physicians NEWS2 algorithm.

---

## 📁 Repository Structure

```
├── notebooks/                     # Interactive Jupyter Notebooks (LinuxONE Jupyter Lab)
│   ├── 01_clinical_data_pipeline.ipynb
│   └── 02_model_training_onnx_benchmark.ipynb
├── pipeline/                      # Telemetry & clinical scoring pipeline
│   ├── __init__.py
│   ├── news2_engine.py           # Royal College of Physicians NEWS2 algorithm
│   ├── vitals_generator.py       # MIMIC-IV streaming biometrics generator
│   ├── dispatch_generator.py     # Ambulance GPS & route simulator
│   └── stream_manager.py         # 30-second rolling window buffer
├── server/                        # Production FastAPI inference engine on s390x
│   └── app.py
├── dashboard/                     # Real-time Streamlit command center
│   └── app.py
├── screenshots/                   # Verification screenshots on LinuxONE
│   ├── 01_linuxone_terminal_specs.png
│   └── 02_jupyterlab_dashboard.png
└── tests/                         # Unit tests
    └── test_pipeline.py
```
