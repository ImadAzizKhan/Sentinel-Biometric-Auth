# SENTINEL: Multimodal Biometric Verification System 🛡️

**Authors:** Muhammad Imad Aziz Khan, Airad Khan, Jassahib Singh  
**Course:** Biometric Systems, Sapienza University of Rome  
**Framework:** ISO/IEC 30107-3 (PAD) & ISO/IEC 19795-1 Standards

---

## 📜 Overview
Sentinel is a high-security multimodal authentication suite that fuses **Facial Recognition (LBPH)** with **Behavioral Keystroke Dynamics**. The system is specifically engineered to mitigate **Presentation Attacks (Spoofing)** through a multi-layered verification pipeline and a strict security-first fusion policy.


---

## 🚀 Features
* **1:1 Verification:** Secure identity matching against stored templates.
* **Advanced Liveness Detection:** Employs **Laplacian Variance** (Spatial) and **LBP Histogram shifts** (Temporal) to block 2D photo and screen-based attacks.
* **Enhanced Keystroke Dynamics:** Utilizes **Z-Score Normalization** and **Mahalanobis Distance** to create a unique statistical "muscle memory" fingerprint.
* **Adaptive Fusion:** Dynamically adjusts weights ($W_{face}$ and $W_{key}$) based on real-time **Image Quality** assessments.
* **Security Veto (Hard-Fail Gate):** Implements a decision-level override; if facial confidence falls below **20%**, access is denied regardless of keystroke performance.
* **Admin Panel:** Features real-time **DET (Detection Error Trade-off) Curve** generation, EER calculation, and intruder snapshot monitoring.


---

## 🛠️ Installation
1.  **Python Environment:** Install Python 3.8+.
2.  **Install Dependencies:**
    ```bash
    pip install opencv-contrib-python numpy keyboard pillow joblib pyttsx3
    ```
3.  **Haar Cascades:** Ensure `haarcascade_frontalface_default.xml` is located in the project root.
4.  **Run System:**
    ```bash
    python Final9.0.py
    ```

---

## 🧪 Testing Protocol (Performance Evaluation)
To generate the metrics required for the project report, follow this standardized protocol:

* **Genuine Test:** Log in normally. On the success screen, click **"✅ YES (Genuine)"** to record correct verification.
* **Spoof Test:** Present a high-quality photo or screen to the camera.
    * If blocked, click **"✅ Correct Reject (Photo)"** to verify the liveness check (APCER stays at 0%).
    * If bypassed, click **"💀 NO (Spoof Passed)"** to record an **APCER** hit.
* **Imposter Test:** Have a different person attempt to log in as a target user. If incorrectly accepted, click **"⚠️ NO (Wrong Person/FAR)"** to record a **FAR** hit.
* **Statistical Integrity:** The system increments the attempt denominator for **every** interaction to ensure transparent reporting.

---

## 🔬 Scientific Methodology
Sentinel utilizes **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for illumination normalization during both enrollment and live recognition. This ensures that the **LBPH** texture patterns remain robust under varying light sources, significantly reducing the False Rejection Rate (FRR) in real-world environments.
