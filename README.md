# EV Battery Fault Diagnosis (Replication Study)

## Reference Paper


S. Yang, Y. Wang, and X. Zhang,  
“Data-driven lithium-ion battery fault diagnosis based on machine learning,”  
Applied Energy, 2020.  



---

## Project Purpose

This project replicates the **data-driven machine learning-based fault diagnosis approach** described in the reference paper and applies it to an Electric Vehicle (EV) battery telemetry dataset.

The goal is to implement supervised classification methods for detecting abnormal operating conditions (Normal vs Warning) using battery and drivetrain telemetry signals.

---

## Dataset Description

File: `EV_Battery_Fault_Diagnosis.csv`

Telemetry features:

- Voltage (V)
- Current (A)
- Temperature (°C)
- Motor Speed (RPM)
- Hall Code
- Estimated SOC (%)
- Ground Truth SOC (%)
- Residual (%)
- Fault Label

Label mapping:
- Normal → 0
- Warning → 1

---

## Important Note on Feature Leakage

The feature **Residual (%)** represents the difference between estimated SOC and ground truth SOC.

Including this feature may cause **label leakage**, since it is strongly correlated with the fault label.

Therefore, experiments are conducted **without the Residual feature** to ensure realistic model evaluation.

---

## Implemented Methods

The following baseline machine learning models are implemented:

- Gaussian Naive Bayes
- Support Vector Machine (RBF kernel)
- Random Forest (300 trees)
- Mahalanobis distance-based anomaly scoring

---

## Evaluation

The script performs:

- Train-test split (75% – 25%, stratified)
- Precision, Recall, F1-score computation
- Confusion matrix generation
- PCA-based 2D visualization

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
Run the script:

```bash
python main.py
