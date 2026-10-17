# 3-Minute Video Pitch Script
### Project: Sub-Millisecond IBM Z Real-Time AI Triage & Emergency Response System
**Team**: Data Enthusiasts  
**Track**: Real-Time AI for Critical Decisions  
**Target Architecture**: IBM z16 / LinuxONE (Telum AI Accelerator)  
**Submission Rule**: Video must be strictly under 3 minutes (180 seconds).

---

## ⏱️ Video Timeline & Screen Breakdown

```
0:00 ───[ 0:35 ]─── Hook & Critical Healthcare Problem
0:35 ───[ 1:15 ]─── IBM Z Architecture & Technical Innovation
1:15 ───[ 2:10 ]─── Live Demonstration (Streamlit + LinuxONE Backend)
2:10 ───[ 2:40 ]─── Mainframe Proof & Sub-Millisecond Latency Benchmark
2:40 ───[ 3:00 ]─── Impact on Emergency Medicine & Wrap-up
```

---

### Segment 1: Hook & Problem Statement (0:00 - 0:35)
* **What to Show on Screen**: Title slide + Emergency room/ambulance graphic or problem statistics slide.
* **Speaker Script**:
  > *"Every year, millions of emergency medical patients suffer irreversible deterioration during transit before reaching the hospital. Today’s telemedicine and cloud AI solutions face two fatal flaws: first, public cellular networks introduce 100 to 500 milliseconds of latency and unpredictable jitter, making real-time vital analysis unsafe for split-second decisions. Second, transmitting continuous patient biometric telemetry across public clouds creates severe HIPAA compliance and cybersecurity vulnerabilities. When a patient is in septic shock or cardiac distress, doctors cannot afford network lag. They need sub-millisecond intelligence."*

---

### Segment 2: Proposed Solution on IBM Z (0:35 - 1:15)
* **What to Show on Screen**: System Architecture diagram from README.
* **Speaker Script**:
  > *"We built the Sub-Millisecond IBM Z Real-Time AI Triage System, deployed natively on the IBM LinuxONE Community Cloud. By co-locating transactional patient telemetry and deep learning inferencing on the IBM Z mainframe, we eliminate the network hop entirely. Our pipeline ingests streaming biometrics calibrated to clinical MIMIC-IV distributions, computes rolling 30-second physiological moving averages, and executes an optimized deep neural network exported to ONNX format directly on the s390x processor architecture with hardware-assisted Telum acceleration."*

---

### Segment 3: Live System Demonstration (1:15 - 2:10)
* **What to Show on Screen**: Screen recording of the **Streamlit Dashboard** (`http://localhost:8501`).
* **Actions on Screen**:
  1. Show the top status banner proving live connection to `http://148.100.108.131:8000` on `s390x`.
  2. Start on *Stable Patient Transit* (Green card, NEWS2 score 0).
  3. Change the dropdown scenario to **"Sepsis Shock Crisis (High Risk)"**.
  4. Point out the instant UI transformation:
     - NEWS2 jumps to 19.
     - **Flashing Red Alert**: *"CRITICAL ACTION TRIGGERED: Immediate emergency critical care team required."*
     - **Hospital Action**: *"ICU Bed #4 Auto-Reserved at Central General Hospital Trauma Bay."*
     - Ambulance icon turns red with sirens active, moving dynamically toward the hospital with real-time ETA.
* **Speaker Script**:
  > *"Here is our live Command Center connected directly to our IBM LinuxONE backend at 148.100.108.131. As our ambulance streams vital signs, a stable patient shows normal ward monitoring. But watch what happens when acute deterioration occurs—here we simulate a sudden septic shock episode. In less than one millisecond, the IBM Z AI engine evaluates the vital signs, calculates a NEWS2 score of 19, triggers an emergency high-risk alarm, and automatically reserves ICU Bed number 4 at the receiving trauma center before the ambulance even arrives. Field paramedics get instant route prioritization with sirens engaged, while the hospital surgical team is pre-notified."*

---

### Segment 4: Mainframe Proof & Latency Benchmark (2:10 - 2:40)
* **What to Show on Screen**: 
  1. Terminal window showing SSH into `linux1@148.100.108.131` with `uname -a`, `lscpu` (pointing out `s390x`, `nnpa`, 5.5 GHz).
  2. The latency comparison bar chart on the dashboard or terminal benchmark (`0.18 ms vs 185 ms`).
* **Speaker Script**:
  > *"This isn't a simulation on x86—our AI microservice and ONNX model run natively on an enterprise IBM LinuxONE instance with 3 vCPUs clocked at 5.5 GHz. In our 1,000-sample empirical benchmark, single-record inference on IBM Z averaged just 184 microseconds—that is 0.18 milliseconds. Compared to a standard public cloud REST API running over cellular networks at 185 milliseconds, IBM LinuxONE delivers over a 1,000x latency advantage."*

---

### Segment 5: Impact & Conclusion (2:40 - 3:00)
* **What to Show on Screen**: Team slide with GitHub repository QR code and team name `[Data Enthusiasts]`.
* **Speaker Script**:
  > *"By combining Royal College of Physicians clinical guidelines, open-source ONNX deep learning, and the raw speed of IBM Z architecture, we turn transit time into treatment time. Thank you to IBM and the Shooting Stars Foundation for empowering Team Data Enthusiasts to build AI that saves lives."*
