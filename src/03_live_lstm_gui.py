import os
import tkinter as tk
from tkinter import font
import serial
import joblib
import pandas as pd
import numpy as np
import threading
import warnings
from collections import deque
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore", category=UserWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
models_dir = os.path.join(root_dir, "Models")

SERIAL_PORT = 'COM10' 
BAUD_RATE = 115200  
SEQUENCE_LENGTH = 10

try:
    # Prefer prefixed filenames for deterministic repo order, fall back to originals
    try:
        model = load_model(os.path.join(models_dir, "04_ev_cooling_lstm.keras"))
    except Exception:
        model = load_model(os.path.join(models_dir, "ev_cooling_lstm.keras"))

    try:
        scaler = joblib.load(os.path.join(models_dir, "04_lstm_scaler.pkl"))
    except Exception:
        scaler = joblib.load(os.path.join(models_dir, "lstm_scaler.pkl"))
except Exception as e:
    print(f"ERROR: Could not load AI files. Run 02_train_lstm.py first! ({e})")
    exit()

# This is the AI's "Short Term Memory"
data_buffer = deque(maxlen=SEQUENCE_LENGTH)

# --- GUI SETUP ---
root = tk.Tk()
root.title("Deep Learning - EV Cooling Diagnostics")
root.geometry("1000x650")
root.configure(bg="#121212") 

title_font = font.Font(family="Helvetica", size=26, weight="bold")
data_font = font.Font(family="Helvetica", size=40, weight="bold")
label_font = font.Font(family="Helvetica", size=18, weight="bold")
rul_font = font.Font(family="Courier New", size=32, weight="bold")

tk.Label(root, text="⚡ AI PREDICTIVE MAINTENANCE ⚡", font=title_font, bg="#121212", fg="#ffffff").pack(pady=10)

data_frame = tk.Frame(root, bg="#121212")
data_frame.pack(pady=10)

rpm_var = tk.StringVar(value="0 RPM")
current_var = tk.StringVar(value="0 mA")

tk.Label(data_frame, textvariable=rpm_var, font=data_font, bg="#121212", fg="#00e5ff").grid(row=0, column=0, padx=40)
tk.Label(data_frame, textvariable=current_var, font=data_font, bg="#121212", fg="#00e5ff").grid(row=0, column=1, padx=40)

# --- AMPS COMPARISON ---
compare_frame = tk.Frame(root, bg="#1a1a1a", bd=1, relief="sunken")
compare_frame.pack(pady=10, fill="x", padx=50)

tk.Label(compare_frame, text="ELECTRICAL LOAD ANALYSIS", font=label_font, bg="#1a1a1a", fg="#00e5ff").pack(pady=5)
diff_var = tk.StringVar(value="BUFFERING AI MEMORY...")
diff_label = tk.Label(compare_frame, textvariable=diff_var, font=("Courier New", 20, "bold"), bg="#1a1a1a", fg="#aaaaaa")
diff_label.pack(pady=5)

# --- STATUS DIAGNOSTICS ---
status_frame = tk.Frame(root, bg="#121212")
status_frame.pack(pady=15)

bearing_var = tk.StringVar(value="BEARING: WAITING...")
airflow_var = tk.StringVar(value="AIRFLOW: WAITING...")

bearing_label = tk.Label(status_frame, textvariable=bearing_var, font=label_font, bg="#121212", fg="#aaaaaa")
bearing_label.grid(row=0, column=0, padx=20)
airflow_label = tk.Label(status_frame, textvariable=airflow_var, font=label_font, bg="#121212", fg="#aaaaaa")
airflow_label.grid(row=0, column=1, padx=20)

# --- RUL FRAME ---
rul_frame = tk.Frame(root, bg="#1a1a1a", bd=2, relief="ridge")
rul_frame.pack(pady=15, ipadx=20, ipady=10)
tk.Label(rul_frame, text="ESTIMATED TIME TO FAILURE (RUL)", font=label_font, bg="#1a1a1a", fg="#aaaaaa").pack()
rul_var = tk.StringVar(value="...")
rul_label = tk.Label(rul_frame, textvariable=rul_var, font=rul_font, bg="#1a1a1a", fg="#ffffff")
rul_label.pack()

# --- THE UPDATE FUNCTION ---
def update_gui(rpm, current, bearing_state, airflow_state):
    rpm_var.set(f"{int(rpm)} RPM")
    current_var.set(f"{int(current)} mA")
    
    healthy_baseline = 275 # <--- 5V Fan baseline
    margin = 55
    
    if rpm < 50:
        diff_var.set("LOAD: NO POWER")
        diff_label.configure(fg="#ff0044")
    elif current > (healthy_baseline + margin):
        diff_var.set(f"LOAD: +{int(current - healthy_baseline)}mA (OVERLOAD)")
        diff_label.configure(fg="#ff0044")
    elif current < (healthy_baseline - margin):
        diff_var.set(f"LOAD: -{int(healthy_baseline - current)}mA (UNDERLOAD)")
        diff_label.configure(fg="#ffaa00")
    else:
        diff_var.set(f"LOAD: OPTIMAL (±{healthy_baseline}mA)")
        diff_label.configure(fg="#00ff00")

    # Hardcoded Stall Override
    if rpm < 50: 
        bearing_var.set("SYSTEM: 🚨 STALLED / JAMMED")
        bearing_label.configure(fg="#ff0044")
        airflow_var.set("AIRFLOW: ---")
        airflow_label.configure(fg="#aaaaaa")
        rul_var.set("0 SECONDS (CRITICAL)")
        rul_label.configure(fg="#ff0044")
        return 

    # LSTM Network Diagnostics
    if airflow_state == 2:
        airflow_var.set("AIRFLOW: 🚨 CRITICAL VACUUM")
        airflow_label.configure(fg="#ff0044")
        rul_var.set("< 15 MINUTES (OVERHEAT)")
        rul_label.configure(fg="#ff0044")
    elif airflow_state == 3:
        airflow_var.set("AIRFLOW: 🚨 CHOKED / BACKPRESSURE")
        airflow_label.configure(fg="#ff0044")
        rul_var.set("< 5 MINUTES (MOTOR BURN)")
        rul_label.configure(fg="#ff0044")
    elif airflow_state == 1:
        airflow_var.set("AIRFLOW: ⚠️ PARTIAL BLOCK")
        airflow_label.configure(fg="#ffaa00")
        rul_var.set("~ 500 HOURS (SERVICE)")
        rul_label.configure(fg="#ffaa00")
    elif bearing_state == 1:
        bearing_var.set("BEARING: ⚠️ WEAR DETECTED")
        bearing_label.configure(fg="#ff0044")
        if airflow_state == 0:
            airflow_var.set("AIRFLOW: OPEN")
            airflow_label.configure(fg="#00ff00")
        rul_var.set("~ 1,200 HOURS (REPLACE)")
        rul_label.configure(fg="#ff0044")
    else:
        bearing_var.set("BEARING: HEALTHY")
        bearing_label.configure(fg="#00ff00")
        airflow_var.set("AIRFLOW: OPEN")
        airflow_label.configure(fg="#00ff00")
        rul_var.set("~ 50,000 HOURS (OPTIMAL)")
        rul_label.configure(fg="#00ff00")

# --- BACKGROUND USB THREAD ---
def read_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception:
        print(f"Could not connect to {SERIAL_PORT}.")
        return

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = line.split(',')
                if len(data) == 6:
                    time_ms, v, ma, mw, temp, rpm = map(float, data)
                    
                    # Add current reading to the AI's rolling memory buffer
                    data_buffer.append([v, ma, mw, rpm])
                    
                    # Wait until we have 10 frames of data before predicting
                    if len(data_buffer) == SEQUENCE_LENGTH:
                        # 1. Convert memory to DataFrame
                        df_buffer = pd.DataFrame(data_buffer, columns=['Voltage_V', 'Current_mA', 'Power_mW', 'RPM'])
                        
                        # 2. Scale the data (Crucial for Neural Networks)
                        scaled_buffer = scaler.transform(df_buffer)
                        
                        # 3. Reshape for LSTM: (1 sample, 10 timesteps, 4 features)
                        lstm_input = np.array([scaled_buffer])
                        
                        # 4. Predict
                        predictions = model.predict(lstm_input, verbose=0)
                        
                        # 5. Extract multi-head outputs
                        # Bearing uses Sigmoid (Prob > 0.5 is a fault)
                        bearing_pred = 1 if predictions[0][0][0] > 0.5 else 0
                        
                        # Airflow uses Softmax (Returns index of highest probability: 0, 1, 2, or 3)
                        airflow_pred = int(np.argmax(predictions[1][0]))
                        
                        root.after(0, update_gui, rpm, ma, bearing_pred, airflow_pred)
                    else:
                        # Update the raw numbers while waiting for the buffer to fill
                        root.after(0, update_gui, rpm, ma, 0, 0)
                        
        except ValueError:
            continue
        except Exception as e:
            print(f"Serial Error: {e}")
            break

thread = threading.Thread(target=read_serial, daemon=True)
thread.start()
root.mainloop()
