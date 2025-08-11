import serial
import time
import pandas as pd
import numpy as np
import joblib

# Load models and scaler
try:
    vitals_model = joblib.load("D:\\ESP32\\IoT Health Monitor\\ML\\vitals_model_2A.pkl")
    vitals_scaler = joblib.load("D:\\ESP32\\IoT Health Monitor\\ML\\vitals_scaler_2A.pkl")
    ecg_model = joblib.load("D:\\ESP32\\IoT Health Monitor\\ML\\kaggle_ecg_model.pkl")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure the .pkl files are in the specified path.")
    exit()

# Serial config
SERIAL_PORT = "COM6"  # Change to your port
BAUD_RATE = 115200

def parse_line(line):
    """
    Parses a single line of comma-separated sensor data.
    Returns a tuple of (bpm, spo2, temp, ecg) or None if parsing fails.
    """
    try:
        parts = line.strip().split(',')
        if len(parts) != 4:
            return None
        bpm = float(parts[0])
        spo2 = float(parts[1])
        temp = float(parts[2])
        ecg = float(parts[3])
        return bpm, spo2, temp, ecg
    except (ValueError, IndexError):
        # Handles cases where the line is not valid CSV or conversion fails
        return None

def predict_vitals(bpm, spo2, temp):
    """
    Predicts a condition based on BPM, SpO2, and temperature using the vitals model.
    """
    # Prepare DataFrame with correct column names your vitals model expects
    df = pd.DataFrame([{
        "pulse": bpm,
        "body temperature": temp,
        "SpO2": spo2
    }])
    try:
        scaled = vitals_scaler.transform(df)
        pred = vitals_model.predict(scaled)[0]
        return pred
    except Exception as e:
        return f"Vitals prediction error: {e}"

def predict_ecg(ecg_window):
    """
    Predicts a condition from a window of ECG data using the ECG model.
    The model expects a window of 187 features.
    """
    try:
        # Reshape the list of 187 values into a 2D array for the model
        arr = np.array(ecg_window).reshape(1, -1)
        pred = ecg_model.predict(arr)[0]
        return pred
    except Exception as e:
        return f"ECG prediction error: {e}"

def main():
    """
    Main function to handle serial communication and predictions.
    """
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for ESP32 reset
        print("Serial port opened successfully.")
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    # ECG window size must match the model's expected input size
    ECG_WINDOW_SIZE = 187
    ecg_window = []

    print("Listening for data...")
    print("-" * 50)

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parsed = parse_line(line)
                
                if parsed:
                    bpm, spo2, temp, ecg = parsed

                    # Predict vitals condition
                    vitals_condition = predict_vitals(bpm, spo2, temp)

                    # Collect ECG data
                    ecg_window.append(ecg)
                    
                    # Predict ECG condition once the window is full
                    if len(ecg_window) >= ECG_WINDOW_SIZE:
                        ecg_condition = predict_ecg(ecg_window)
                        # Reset the window for the next prediction
                        ecg_window = []
                    else:
                        ecg_condition = f"Collecting ECG data... ({len(ecg_window)}/{ECG_WINDOW_SIZE})"

                    # Print real-time results
                    print(f"Vitals → BPM: {bpm:.2f}, SpO2: {spo2}, Temp: {temp:.2f}°C → Condition: {vitals_condition}")
                    print(f"ECG Health Prediction: {ecg_condition}")
                    print("-" * 50)
                else:
                    # You can uncomment this line for debugging to see any unparsed lines
                    # print(f"Skipping malformed line: {line}")
                    pass
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
