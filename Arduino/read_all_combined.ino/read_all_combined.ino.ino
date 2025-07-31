// ---------------- Required Libraries ----------------
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include <OneWire.h>
#include <DallasTemperature.h>

// ---------------- Sensor Pins ----------------
#define ECG_PIN 34
#define ONE_WIRE_BUS 4 // DS18B20 data pin on GPIO4 (change if needed)

// ---------------- Sensor Objects ----------------
MAX30105 particleSensor;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

// ---------------- Variables ----------------
float temperature = 0.0;
int ecgValue = 0;
int32_t irValue, redValue;

const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute;
int beatAvg;

// ---------------- Setup ----------------
void setup() {
  Serial.begin(115200);
  delay(1000); // Allow time for Serial to initialize

  Serial.println("🔌 Setup starting...");

  // Initialize Temperature Sensor
  tempSensor.begin();
  Serial.println("🌡️ Temperature sensor initialized.");

  // Initialize MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("❌ MAX30102 not found. Check wiring and power supply.");
    while (1); // Halt execution
  }

  particleSensor.setup(); // Use default settings
  particleSensor.setPulseAmplitudeRed(0x0A);  // Low power
  particleSensor.setPulseAmplitudeIR(0x0A);   // Low power
  Serial.println("❤️ MAX30102 initialized.");
}

// ---------------- Main Loop ----------------
void loop() {
  // ----------- MAX30102 Readings -----------
  irValue = particleSensor.getIR();
  redValue = particleSensor.getRed();

  if (checkForBeat(irValue)) {
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 255 && beatsPerMinute > 20) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;

      beatAvg = 0;
      for (byte i = 0; i < RATE_SIZE; i++)
        beatAvg += rates[i];
      beatAvg /= RATE_SIZE;
    }
  }

  // ----------- Temperature Readings -----------
  tempSensor.requestTemperatures();
  temperature = tempSensor.getTempCByIndex(0);

  // ----------- ECG Reading -----------
  ecgValue = analogRead(ECG_PIN);

  // ----------- SpO2 Placeholder -----------
  int spo2 = 0;

  // ----------- Serial Output (CSV Format) -----------
  Serial.print(beatsPerMinute, 2); Serial.print(",");
  Serial.print(spo2); Serial.print(",");
  Serial.print(temperature, 2); Serial.print(",");
  Serial.println(ecgValue);

  delay(100); // Small delay to avoid flooding Serial
}
