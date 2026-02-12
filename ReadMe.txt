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
* **Enhanced Keystroke Dynamics:** Utilizes **Z-Score Normalization** and **Mahalanobis Distance** with a calibrated decay factor to create a unique behavioral "muscle memory" fingerprint.
* **Adaptive Fusion:** Dynamically adjusts weights ($W_{face}$ and $W_{key}$) based on real-time **Image Quality** assessments.
* **Security Veto (Hard-Fail Gate):** Implements a decision-level override; if facial confidence falls below **20%**, access is denied regardless of keystroke performance.
* **Admin Panel:** Features real-time **DET (Detection Error Trade-off) Curve** generation, EER calculation, and intruder snapshot monitoring.



---

## 🧪 Simulation & Dataset Setup
To replicate the experimental results or test with a larger chimeric population, download the following open-source datasets:

1.  **Face Dataset:** [AT&T Database of Faces (Kaggle)](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)
    * Download and extract folders (`s1` through `s40`) into `raw_data/faces/`.
2.  **Keystroke Dataset:** [CMU Keystroke Dynamics (DSL)](http://www.cs.cmu.edu/~keystroke/)
    * Place the `DSL-StrongPasswordData.csv` file into the `raw_data/` folder.

### **Running the Simulation Pipeline**
* **Identity Fusion (`builder.py`):** Run this first. It pairs facial folders with keystroke rows to create "Virtual Users." This script is **dynamic** and scales automatically based on the number of folders provided.
* **Model Training & Evaluation (`create_users.py`):** Run this to train the LBPH recognizer and Random Forest classifier. It applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for illumination consistency.

---

## ⚙️ Core Scripts
* **create_users.py:** Performs an automated train/test split (50/50), trains models, and provides accuracy metrics for the chimeric identities.
* **builder.py:** Handles the bulk conversion of `.pgm` face files to `.jpg` and maps individual keystroke rows to unique user IDs for the "Chimeric" setup.

---

## 🛠️ Installation
1.  **Python Environment:** Install Python 3.8+.
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run System:**
    ```bash
    python Final9.0.py
    ```
---

## 📂 Project Structure
For the simulation and training scripts to function, ensure your local directory is organized as follows:

```text
SENTINEL-BIO-AUTH/
├── raw_data/
│   ├── faces/               # Place AT&T 's1', 's2'... folders here
│   └── DSL-StrongPasswordData.csv  # CMU Keystroke dataset
├── dataset/                 # Generated automatically by builder.py
│   ├── faces/               # Chimeric face templates
│   └── keystrokes/          # Chimeric keystroke CSVs
├── models/                  # Stored .yml models and user mappings
├── Final9.0.py              # Main Application (Live & Simulation)
├── builder.py               # Chimeric Identity Creator Script
├── create_users.py          # Model Evaluator & Trainer Script
└── requirements.txt         # Project dependencies

---

## 🧪 Testing Protocol (Performance Evaluation)
To generate the metrics required for the project report, use the following standardized protocol:

* **Genuine Test:** Log in normally. On the success screen, click **"✅ YES (Genuine)"** to record correct verification.
* **Spoof Test:** Present a high-quality photo or screen to the camera.
    * If blocked, click **"✅ Correct Reject (Photo)"** to verify liveness (APCER stays at 0%).
    * If bypassed, click **"💀 NO (Spoof Passed)"** to record an **APCER** hit.
* **Imposter Test:** Have a different person attempt to log in as a target user. If incorrectly accepted, click **"⚠️ NO (Wrong Person/FAR)"** to record a **FAR** hit.
* **Statistical Integrity:** The system increments the attempt denominator for **every** interaction to ensure transparent reporting.



---

## 🔬 Scientific Methodology
Sentinel utilizes **CLAHE** for pre-processing to ensure the **LBPH** texture patterns remain robust under varying light sources. The keystroke engine uses **Z-Score Normalization** to focus on the keys where a user is naturally consistent, effectively filtering out behavioral noise.
