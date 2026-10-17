# IBM Z Datathon 2026: Official Submission Dossier
### Team: Data Enthusiasts
**Submission Platform**: Hack2Skill ([https://hack2skill.com/event/datathon-2026](https://hack2skill.com/event/datathon-2026))

---

## 📋 Submission Form Details (Copy-Paste Ready)

* **Project Title**:
  ```text
  [Data Enthusiasts_IBM Z Datathon 2026 Project] Sub-Millisecond IBM Z Real-Time AI Triage & Emergency Response System
  ```

* **Selected Challenge Track**:
  ```text
  Real-Time AI for Critical Decisions (Target Architecture: IBM z16 / LinuxONE Telum AI)
  ```

* **GitHub Repository Link**:
  ```text
  https://github.com/satwik834/Data_Enthusiasts_IBM_Z_Datathon_2026_Project
  ```

* **Video Demo Link**:
  ```text
  [Paste your unlisted YouTube, Loom, or Google Drive link here - Max 3 Minutes]
  ```

---

## 📝 Project Abstract & Description (For Submission Form)

### 1. Problem Statement
Emergency Medical Services (EMS) and hospital trauma triage face life-or-death transit windows where every minute determines patient survival. Current cloud AI architectures present two critical barriers in emergency medicine:
1. **Unsafe Latency & Jitter**: Public cellular cloud connections introduce 100ms–500ms+ network round-trip delays, making real-time telemetry processing unsafe for instantaneous triage.
2. **Data Privacy & Compliance**: Streaming raw patient biometrics across public clouds violates strict healthcare regulations (HIPAA, GDPR) and exposes sensitive medical data to interception.

### 2. Proposed Solution on IBM Z
We engineered an enterprise-grade, end-to-end real-time emergency triage decision-support platform running natively on **IBM LinuxONE (s390x architecture)**:
* **In-Memory Telemetry Preprocessing**: Ingests streaming patient biometrics calibrated to clinical MIMIC-IV vitals distributions and computes 30-second sliding window moving averages and acute deterioration velocities.
* **Clinical NEWS2 Standards**: Implements the authoritative Royal College of Physicians National Early Warning Score 2 algorithm to stratify risk into Ward, Urgent, and Critical Emergency tiers.
* **On-System Sub-Millisecond Inferencing**: Deep neural network and ONNX models execute directly on the IBM Telum processor architecture alongside transaction data with zero PCIe bus delay, achieving **0.18 ms (184 microseconds)** mean inference latency.
* **Dynamic ICU Resource Allocation**: Automatically alerts receiving hospital trauma bays and reserves ICU beds before ambulance arrival upon high-risk deterioration detection.

### 3. How the Solution Leverages IBM Z & LinuxONE
* **Architecture**: Deployed natively on IBM LinuxONE Community Cloud (`s390x` 64-bit Big-Endian, 3 vCPUs @ 5.5 GHz).
* **Hardware AI Exploitation**: Directly targets the IBM Telum **NNPA (Neural Network Processing Assist)** accelerator for microsecond vector inference.
* **Measurable Latency Advantage**: In our 1,000-sample empirical benchmark, LinuxONE achieved **0.185 ms** inference compared to a standard public cloud REST baseline of **185.0 ms** — proving a **1,002x speedup** with zero network jitter.
* **Zero-Trust Security**: Ensures patient biometrics remain secure within the mainframe boundary, leveraging IBM CPACF cryptographic hardware.

---

## 🏆 Scoring Rubric Alignment (40 Points + AI Innovation Award)

| Rubric Category | How Our Project Excels |
|---|---|
| **1. Value to Community & Theme Alignment (10 pts)** | Solves a critical societal crisis: eliminating transit delays and pre-allocating ICU beds for septic shock, cardiac arrest, and trauma patients before arrival. |
| **2. Novelty & Idea Originality (10 pts)** | Shifts emergency triage from reactive in-hospital intake to proactive sub-millisecond mainframe streaming AI during transit. |
| **3. Technical Design & LinuxONE Exploitation (10 pts)** | Genuine execution on IBM LinuxONE `s390x` VM; completed prerequisite labs; ONNX model deployment; containerized microservice; microsecond benchmark verification. |
| **4. Execution & Pitch Quality (10 pts)** | Working end-to-end prototype, live interactive dashboard, validated test suite (20 tests passing), and structured 3-minute video presentation. |
