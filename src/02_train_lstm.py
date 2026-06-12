import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
data_dir = os.path.join(root_dir, "Data")
models_dir = os.path.join(root_dir, "Models")

os.makedirs(models_dir, exist_ok=True)

SEQUENCE_LENGTH = 10  # The AI will memorize the last 10 data points

print("1. Loading Master Dataset...")
# Updated filename to prefixed ordering
df = pd.read_csv(os.path.join(data_dir, "05_master_ev_cooling_dataset.csv"))

print("2. Scaling the Hardware Data...")
features = ['Voltage_V', 'Current_mA', 'Power_mW', 'RPM']

# Neural networks need scaled data. We save this scaler for the live GUI!
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, os.path.join(models_dir, "lstm_scaler.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "04_lstm_scaler.pkl"))

print("3. Chopping data into time sequences...")
def create_sequences(data, bearing_labels, airflow_labels, seq_length):
    X, y_bearing, y_airflow = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:(i + seq_length)].values)
        # The label is the state at the END of the window
        y_bearing.append(bearing_labels.iloc[i + seq_length])
        y_airflow.append(airflow_labels.iloc[i + seq_length])
    return np.array(X), np.array(y_bearing), np.array(y_airflow)

X_seq, y_b_seq, y_a_seq = create_sequences(df[features], df['Bearing_Fault'], df['Airflow_Fault'], SEQUENCE_LENGTH)

X_train, X_test, y_b_train, y_b_test, y_a_train, y_a_test = train_test_split(
    X_seq, y_b_seq, y_a_seq, test_size=0.2, random_state=42
)

print("\n4. Forging the Multi-Head LSTM Architecture...")
inputs = Input(shape=(SEQUENCE_LENGTH, len(features)))

# The Deep Learning "Memory" Core
x = LSTM(64, return_sequences=False)(inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)

# Head 1: Predicts Bearing Wear (0 to 1 Probability)
out_bearing = Dense(1, activation='sigmoid', name='bearing_output')(x)
# Head 2: Predicts Airflow State (0, 1, 2, or 3)
out_airflow = Dense(4, activation='softmax', name='airflow_output')(x)

model = Model(inputs=inputs, outputs=[out_bearing, out_airflow])

model.compile(
    optimizer='adam',
    loss={'bearing_output': 'binary_crossentropy', 'airflow_output': 'sparse_categorical_crossentropy'},
    metrics={'bearing_output': 'accuracy', 'airflow_output': 'accuracy'}
)

print("\n5. Training the Deep Neural Network... (This may take a minute!)")
model.fit(
    X_train, {'bearing_output': y_b_train, 'airflow_output': y_a_train},
    validation_data=(X_test, {'bearing_output': y_b_test, 'airflow_output': y_a_test}),
    epochs=15, batch_size=32
)

print("\n6. Saving the AI to your hard drive...")
# Keep model filename as original so existing trained model remains compatible
model.save(os.path.join(models_dir, "ev_cooling_lstm.keras"))
# Also save prefixed copy for deterministic GitHub ordering
model.save(os.path.join(models_dir, "04_ev_cooling_lstm.keras"))
print("\nSUCCESS! Saved model files and scaler (includes prefixed copies if supported).")
