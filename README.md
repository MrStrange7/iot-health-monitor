# IoT Health Monitor using ESP32 + Machine Learning
A smart health monitoring system that collects real-time data from wearable sensors using an ESP32, logs it, and predicts health conditions like Fever, Hypoxia, and general Unwellness using Machine Learning.



## 📊 Features
Real-time data collection from sensors:


- MAX30102 – Heart Rate & SpO₂
- DS18B20 – Body Temperature
- AD8232 – ECG Monitoring
- Serial communication to send sensor data to a laptop
- Machine Learning predictions using:
  - Random Forest Classifier (primary)
  - Decision Tree (alternative)
- Logs sensor readings and predictions to CSV
- Easily extendable with Blynk, OLED display, or emergency alerts

## 📂 Folder Structure

```
IoT-Health-Monitor/
│
├── Arduino/
│ ├── read_ad8232.ino
│ ├── read_ds18b20.ino
│ ├── read_max30102.ino
│ └── read_all_combined.ino
│
├── ML/
│ ├── health_data.csv # Sample dataset
│ ├── health_model.pkl # Trained Decision Tree model
│ ├── health_rf_model.pkl # Trained Random Forest model
│ ├── health_model_training.ipynb # Jupyter Notebook for training
│ ├── train_health_model.py # Python script to train model
│ └── log_to_csv.py # Log real-time sensor data
│
├── simulate_esp32.py # Fake serial data generator for testing
├── serial_listener.py # Receives serial data & predicts health
├── health_predictor.py # Model loading & prediction utilities
└── requirements.txt # Python dependencies
```
## Getting Started
1️⃣  Setup Python Environment
```
pip install -r ML/requirements.txt
```
2️⃣  Train the Model (Optional)
Using Jupyter Notebook: ML/health_model_training.ipynb

Using Python script:

```
python ML/train_health_model.py
```
3️⃣  Upload the desired Arduino sketch to your ESP32 using the Arduino IDE before running the Python scripts.

4️⃣ Test / Simulate Data
Simulate ESP32 data without hardware:
```
python simulate_esp32.py
```
Use with real ESP32 data via USB/Serial:

```
python serial_listener.py
```
## Hardware Components
- ESP32 microcontroller
- MAX30102 (Heart Rate & SpO₂)
- DS18B20 (Temperature sensor)
- AD8232 (ECG sensor)
- Jumper wires, breadboard
- OLED display (optional)
- USB cable / Power supply

## How to Contribute
Clone the repo:
```
git clone https://github.com/MrStrange7/iot-health-monitor.git
```
Fork it and submit pull requests for improvements

## License
This project is licensed under the MIT License.
