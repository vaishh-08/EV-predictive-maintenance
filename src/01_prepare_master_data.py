import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
data_dir = os.path.join(root_dir, "Data")

# 1. FIXED THE FILENAME (old_fan_blocked instead of full)
files_to_load = [
    (os.path.join(data_dir, "06_new_fan_open.csv"), 0, 0),
    (os.path.join(data_dir, "08_new_fan_partial.csv"), 0, 1),
    (os.path.join(data_dir, "07_new_fan_full.csv"), 0, 2),
    (os.path.join(data_dir, "10_old_fan_open.csv"), 1, 0),
    (os.path.join(data_dir, "11_old_fan_partial.csv"), 1, 1),
    (os.path.join(data_dir, "09_old_fan_blocked.csv"), 1, 2) # <--- Fixed!
]

all_dataframes = []
print("Starting Data Stitching Process...")

# The exact 6 columns your ESP32 spits out
EXPECTED_COLUMNS = ['Time_ms', 'Voltage_V', 'Current_mA', 'Power_mW', 'Temp_C', 'RPM']

for filename, bearing_label, airflow_label in files_to_load:
    if os.path.exists(filename):
        print(f"Loading {filename}...")
        try:
            # Read the CSV
            df = pd.read_csv(filename)
            
            # Force it to only keep the first 6 columns (drops accidental empty columns)
            df = df.iloc[:, :6] 
            
            # Force the column names to be perfectly identical
            df.columns = EXPECTED_COLUMNS
            
            # Add the physical state labels
            df['Bearing_Fault'] = bearing_label
            df['Airflow_Fault'] = airflow_label
            
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error cleaning {filename}: {e}")
    else:
        print(f"ERROR: Could not find {filename}!")

if all_dataframes:
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Clean out any rows where the text headers accidentally got mixed into the numbers
    master_df = master_df[pd.to_numeric(master_df['RPM'], errors='coerce').notnull()]
    
    master_df.to_csv(os.path.join(data_dir, "05_master_ev_cooling_dataset.csv"), index=False)
    print(f"\nSUCCESS! Created a clean, perfect master dataset with {len(master_df)} rows.")
else:
    print("\nFAILED: No files were loaded.")
