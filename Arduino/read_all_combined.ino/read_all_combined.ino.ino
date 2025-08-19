#include <Wire.h>
#include "MAX30105.h"
#include <OneWire.h>
#include <DallasTemperature.h>

#define ONE_WIRE_BUS 5     // DS18B20 data pin
#define ECG_PIN 34         // AD8232 analog output pin

MAX30105 particleSensor;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found. Check wiring.");
    while (1);
  }
  particleSensor.setup(); // Use default settings

  // Initialize DS18B20
  sensors.begin();

  Serial.println("ESP32 Health Monitor Started");
}

void loop() {
  // === MAX30102 readings ===
  long irValue = particleSensor.getIR(); // Raw IR for heart rate calculation
  long redValue = particleSensor.getRed();

  float bpm = particleSensor.getHeartRate();   // Optional, or calculate manually
  float spo2 = particleSensor.getSpO2();       // Optional, or set placeholder

  // === DS18B20 reading ===
  sensors.requestTemperatures();
  float temperature = sensors.getTempCByIndex(0);

  // === AD8232 ECG reading ===
  int ecgValue = analogRead(ECG_PIN);

  // === Send CSV data via Serial ===
  Serial.print(bpm);
  Serial.print(",");
  Serial.print(spo2);
  Serial.print(",");
  Serial.print(temperature);
  Serial.print(",");
  Serial.println(ecgValue);

  delay(500); // Read every 500ms
}
