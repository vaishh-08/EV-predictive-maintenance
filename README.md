# ⚡ Edge AI Predictive Maintenance for EV Thermal Systems

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange.svg)
![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20Arduino-lightgrey.svg)
![UI](https://img.shields.io/badge/UI-CustomTkinter-brightgreen.svg)

An end-to-end predictive maintenance system designed to protect Electric Vehicle (EV) auxiliary cooling components. This project deploys a Long Short-Term Memory (LSTM) neural network at the Edge to diagnose mechanical bearing degradation and aerodynamic blockages using **only** the motor's 4-dimensional electrical footprint—eliminating the need for bulky, expensive external vibration sensors.

![Dashboard Preview](assets/dashboard_ui.png) ---

## 🚀 Core Innovations

* **Non-Invasive Diagnostics:** Analyzes high-speed Voltage (V), Current (mA), Power (mW), and Motor Speed (RPM) telemetry via I2C sensors.
* **Deep Learning Edge AI:** An LSTM network processes a 10-step rolling sequence buffer to detect the temporal current ripple and magnetic pulsing indicative of a struggling motor.
* **Zero Data Leakage:** Ambient temperature is monitored for human operators but explicitly excluded from the AI tensor. This forces the network to learn core electrical impedance rather than "cheating" by memorizing thermal symptoms.
* **State-Based RUL Mapping:** Replaces error-prone regression with deterministic state mapping. Active bearing wear is mapped to the P-F (Potential-to-Failure) curve, providing a reliable 1,200-hour replacement warning window.
* **Live Telemetry Dashboard:** A multi-threaded CustomTkinter GUI featuring real-time electrical load analysis and AI state classification.

---

## 📂 Repository Structure

The project follows a modular, production-ready architecture separating source code, raw data, and compiled models.

```text
EV-Predictive-Maintenance/
│
├── Data/                   # Raw sensor telemetry and training datasets
│   ├── 05_master_ev_cooling_dataset.csv
│   └── (Additional fan state CSVs)
│
├── Models/                 # Compiled AI brains and scalers
│   ├── ev_cooling_lstm.keras    # The trained multi-head LSTM model
│   └── lstm_scaler.pkl          # StandardScaler for data normalization
│
├── src/                    # Core Python application logic
│   ├── 01_prepare_master_data.py  # Data cleaning and aggregation pipeline
│   ├── 02_train_lstm.py           # Neural network architecture and training script
│   └── 03_live_lstm_gui.py        # The Edge dashboard and serial telemetry reader
│
├── .gitignore              # Ignores __pycache__ and system files
└── README.md               # Project documentation

---

## 📊 Dataset Note

Due to GitHub's file size constraints and best practices for repository management, the CSV files located in the `/Data` directory represent a **condensed sample** of the raw telemetry data. 

* The included sample dataset is fully functional and perfectly formatted to allow anyone to run the Edge dashboard and test the LSTM model locally.
* The complete, uncompressed run-to-failure dataset (which spans many hours of high-frequency sensor readings) is retained offline. 

---

🧠 Machine Learning Architecture

The system transitions from standard snapshot-based classification (e.g., Random Forest) to a Time-Series Deep Learning model to capture dynamic mechanical degradation.

Model Type: Long Short-Term Memory (LSTM) Neural Network

Input Features: [Voltage_V, Current_mA, Power_mW, RPM] (Normalized via StandardScaler)

Temporal Window: 10 sequential time-steps

Multi-Head Output: * Mechanical Head (Sigmoid Activation): Binary classification (Healthy vs. Bearing Degradation)

Aerodynamic Head (Softmax Activation): Multi-class categorization (Open Airflow vs. Partial Block vs. Critical Choke)

🛠️ Hardware Setup
Edge Data Acquisition: ESP32 / Arduino Microcontroller

Actuator: 5V BLDC Cooling Fan (EV Thermal Emulator)

Sensors: INA219(current and Voltage ) , DS18B20 Temp sensor

Communication: 115200-baud isolated serial telemetry

💻 Installation & Setup
1. Clone the Repository
Bash
git clone [https://github.com/YourUsername/EV-Predictive-Maintenance.git](https://github.com/YourUsername/EV-Predictive-Maintenance.git)
cd EV-Predictive-Maintenance
2. Install Dependencies
Ensure you have Python 3.9+ installed. Install the required libraries using pip:

Bash
pip install tensorflow pandas numpy customtkinter pyserial joblib scikit-learn

3. Connect Hardware
Flash your microcontroller with your telemetry data-logger code.

Connect the microcontroller to your PC via USB.

Open src/03_live_lstm_gui.py and verify the SERIAL_PORT variable matches your active port (e.g., COM10 or /dev/ttyUSB0).

4. Run the Edge Dashboard
Navigate into the source folder and launch the UI:

Bash
cd src
python 03_live_lstm_gui.py
⚠️ Notes for Developers
Relative Paths: The Python scripts inside /src use relative paths to access /Models and /Data. Ensure you execute the scripts from within the /src directory to prevent path errors.

Arduino Serial Monitor: Ensure the Arduino IDE Serial Monitor is closed before running the Python dashboard, otherwise, a PermissionError will occur when Python attempts to read the COM port.