import serial
import time
import pandas as pd
import joblib  # For loading saved ML models
from health_predictor import predict_health  # Your utility function

# === CONFIG ===
SERIAL_PORT = 'COM6'  # Change this to your ESP32 serial port
BAUD_RATE = 115200
CSV_LOG_FILE = 'realtime_log.csv'
MODEL_PATH = 'ML/health_rf_model.pkl'  # Using Random Forest

# Load trained ML model
model = joblib.load(MODEL_PATH)

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException:
    print(f"Error: Could not open serial port {SERIAL_PORT}")
    exit(1)

# Initialize CSV logging
columns = ['BPM', 'SpO2', 'Temperature', 'ECG', 'Prediction']
try:
    df = pd.read_csv(CSV_LOG_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
    df.to_csv(CSV_LOG_FILE, index=False)

print("Listening for data... Press Ctrl+C to stop.")

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                # Assuming comma-separated values: BPM,SpO2,Temp,ECG
                values = [float(x) for x in line.split(',')]
                if len(values) != 4:
                    print(f"Unexpected data format: {line}")
                    continue

                bpm, spo2, temp, ecg = values

                # Prepare dataframe row for prediction
                input_df = pd.DataFrame([values], columns=['BPM', 'SpO2', 'Temperature', 'ECG'])
                prediction = predict_health(model, input_df)[0]

                # Append to log
                df.loc[len(df)] = [bpm, spo2, temp, ecg, prediction]
                df.to_csv(CSV_LOG_FILE, index=False)

                print(f"BPM: {bpm}, SpO2: {spo2}, Temp: {temp}, ECG: {ecg}, Prediction: {prediction}")

            except ValueError:
                print(f"Error parsing line: {line}")

        time.sleep(0.1)  # Slight delay to avoid overwhelming CPU

except KeyboardInterrupt:
    print("\nStopped by user.")
    ser.close()
