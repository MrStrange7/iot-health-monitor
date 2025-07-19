import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# === CONFIGURATION ===
NUM_SAMPLES = 1000            # Total ECG records
SAMPLE_LENGTH = 100           # Number of voltage readings per sample (e.g., 10s ECG signal @ 100ms)

# === ECG SIGNAL SIMULATORS ===
def generate_healthy_ecg(length):
    # Simulate a sine wave with slight noise to mimic a regular heartbeat
    x = np.linspace(0, 2 * np.pi, length)
    ecg = np.sin(5 * x) * 0.5 + np.random.normal(0, 0.05, length)
    return ecg

def generate_unhealthy_ecg(length):
    # Simulate a noisy or flatline ECG
    choice = random.choice(["flat", "irregular"])
    if choice == "flat":
        ecg = np.random.normal(0.02, 0.02, length)
    else:
        ecg = np.sin(5 * np.linspace(0, 2 * np.pi, length)) * 0.5
        noise = np.random.normal(0, 0.25, length) * np.random.choice([1, -1], length)
        ecg += noise
    return ecg

# === GENERATE DATA ===
data = []
labels = []

for _ in range(NUM_SAMPLES // 2):
    healthy = generate_healthy_ecg(SAMPLE_LENGTH)
    unhealthy = generate_unhealthy_ecg(SAMPLE_LENGTH)
    data.append(healthy)
    labels.append("Healthy")
    data.append(unhealthy)
    labels.append("Unhealthy")

# === CREATE CSV ===
df = pd.DataFrame(data)
df["Label"] = labels
csv_path = "synthetic_ecg_data.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Synthetic ECG dataset saved to {csv_path}")

# === OPTIONAL: Plot example ===
plt.plot(data[0], label="Healthy")
plt.plot(data[1], label="Unhealthy")
plt.title("Example Synthetic ECG Signals")
plt.xlabel("Time (samples)")
plt.ylabel("Voltage (normalized)")
plt.legend()
plt.grid(True)
plt.show()
