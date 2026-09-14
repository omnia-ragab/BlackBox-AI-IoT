#  BlackBox: AI-Powered Predictive Maintenance for Legacy Industrial Equipment

> **SMARTX Hackathon 2026 Submission**  
> *Non-invasive, affordable, and locally-supported predictive maintenance platform built specifically for manufacturing SMEs.*

---

##  Executive Summary
70% of Egyptian and regional factories operate legacy machinery installed two to three decades ago without embedded sensors or IoT capabilities. When these machines fail unexpectedly, manufacturers lose between **100,000 to 400,000 EGP** per incident. 

**BlackBox** solves this by offering a non-invasive IoT retrofit solution paired with a cloud-based AI anomaly detection engine. It alerts operators weeks before a failure occurs—all at **1/15th the cost** of global enterprise solutions and with a **1-week deployment time**.

---

## Key Features
- **Non-Invasive IoT Simulation:** Simulates real-time sensor streams (3-axis vibration, temperature, and acoustic signals) mimicking physical ESP32 hardware modules[cite: 1].
- **AI-Driven Anomaly Detection:** Utilizes machine learning models (**Isolation Forest**) trained on industrial bearing degradation datasets to detect early mechanical wear with high precision and low false-positive rates[cite: 1].
- **Interactive Real-Time Dashboard:** Built with **Streamlit** and **Plotly** to visualize sensor metrics, monitor health statuses, and track anomaly scores instantly[cite: 1].
- **Cost-Benefit Analysis Integration:** Automatically calculates the financial impact and cost differences between emergency failures and planned, prevented maintenance[cite: 1].

---

## Tech Stack
- **Programming Language:** Python[cite: 1]
- **Machine Learning:** Scikit-Learn (Isolation Forest)[cite: 1]
- **Data Processing:** Pandas, NumPy, SciPy[cite: 1]
- **Backend & Simulation:** FastAPI, Custom Python Simulation Scripts[cite: 1]
- **Frontend & Visualization:** Streamlit, Plotly[cite: 1]
- **Model Persistence:** Joblib[cite: 1]

---

##  Repository Structure
```text
├── app.py                      # Interactive Streamlit Dashboard & UI
├── sensor_simulator.py         # IoT Sensor Data Stream Simulator (ESP32 emulation)
├── test_api.py                 # API and backend route testing script
├── blackbox_isolation_forest.joblib # Pre-trained machine learning model
├── blackbox_features_scored.csv     # Processed dataset with anomaly scores
└── live_simulation.png         # Live monitoring and prediction visualization
