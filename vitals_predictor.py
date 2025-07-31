import joblib
import numpy as np

# === LOAD ECG MODEL ===
try:
    ecg_model = joblib.load("ML/kaggle_ecg_model.pkl")
    print("✅ ECG model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load ECG model: {e}")
    ecg_model = None

# === LABEL MAP ===
label_map = {
    0: "Normal",
    1: "Abnormal"
}

# === ECG PREDICTOR FUNCTION ===
def predict_ecg_condition(ecg_signal):
    """
    Predicts heart condition using the ECG signal.
    Expects ecg_signal as a list of 100 float values.
    """
    if ecg_model is None:
        return "Model not loaded"

    try:
        signal_array = np.array(ecg_signal).reshape(1, -1)
        prediction = ecg_model.predict(signal_array)[0]
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        print(f"[ECG Prediction Error] {e}")
        return "Prediction Error"
