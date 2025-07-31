import serial
import time
import joblib
import pandas as pd
import numpy as np
from vitals_predictor import predict_ecg_condition

# === CONFIGURATION ===
SERIAL_PORT = "COM3"         # 🛠️ Replace with your ESP32 port
BAUD_RATE = 9600
MODEL_PATH = "ML/vitals_model_2A.pkl"
SCALER_PATH = "ML/vitals_scaler_2A.pkl"

# === LOAD THE MODEL AND SCALER ===
try:
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Loaded vitals model and scaler.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    exit()

# === HELPER FUNCTION ===
def parse_sensor_data(line):
    try:
        parts = line.strip().split(',')
        if len(parts) != 4:
            raise ValueError("Expected 4 values")
        hr = float(parts[0])
        spo2 = float(parts[1])
        temp = float(parts[2])
        ecg = float(parts[3])
        return {
            "HeartRate": hr,
            "SpO2": spo2,
            "Temperature": temp,
            "ECG": ecg
        }
    except Exception as e:
        print(f"[Parse Error] {e} → Raw line: {line}")
        return None

# === MAIN LISTENER ===
def main():
    print("🔌 Listening for data from ESP32...\n")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Allow time for ESP32 reset
    except Exception as e:
        print(f"❌ Could not open serial port {SERIAL_PORT}: {e}")
        return

    ecg_window = []

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8").strip()
                data = parse_sensor_data(line)

                if data:
                    input_df = pd.DataFrame([data])
                    scaled_input = scaler.transform(input_df)

                    prediction = rf_model.predict(scaled_input)[0]

                    label_map = {0: "Healthy", 1: "Fever", 2: "Hypoxia", 3: "Unwell"}
                    condition = label_map.get(prediction, "Unknown")

                    print(f"📟 HR={data['HeartRate']}, SpO₂={data['SpO2']}, Temp={data['Temperature']}, ECG={data['ECG']} → Vitals Condition: {condition}")

                    # → ECG Signal Prediction (using 100-point window)
                    ecg_window.append(data["ECG"])
                    if len(ecg_window) == 100:
                        ecg_prediction = predict_ecg_condition(ecg_window)
                        print(f"❤️ ECG Health Prediction: {ecg_prediction}")
                        ecg_window.clear()

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"[Runtime Error] {e}")

if __name__ == "__main__":
    main()
