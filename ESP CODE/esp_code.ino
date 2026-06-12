#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// --- PIN DEFINITIONS ---
#define RPM_PIN 18         // The Yellow/White 3rd wire from the 5V Fan
#define ONE_WIRE_BUS 4     // The Yellow data wire from the DS18B20 Temp Sensor

// --- SENSOR OBJECTS ---
Adafruit_INA219 ina219;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);

// --- RPM VARIABLES ---
// We use volatile because these are changed inside an interrupt
volatile unsigned int pulseCount = 0; 
unsigned int rpm = 0;
unsigned long lastMillis = 0;

// --- INTERRUPT SERVICE ROUTINE (ISR) ---
// This tiny function runs in the background every time the fan blade completes a rotation
void IRAM_ATTR countPulse() {
  pulseCount++;
}

void setup() {
  // Start the USB connection at high speed
  Serial.begin(115200);
  while (!Serial) { delay(1); } // Wait for serial connection
  
  // 1. Initialize INA219 Power Sensor
  if (!ina219.begin()) {
    Serial.println("Failed to find INA219 chip. Check I2C wiring (SDA=21, SCL=22)!");
    while (1) { delay(10); }
  }
  // Optional: Calibrate for 32V and 1A for higher precision on a 5V system
  ina219.setCalibration_32V_1A();

  // 2. Initialize DS18B20 Temp Sensor
  tempSensor.begin();

  // 3. Initialize Fan RPM Pin
  // We use INPUT_PULLUP so we don't need a physical resistor for the tachometer
  pinMode(RPM_PIN, INPUT_PULLUP); 
  
  // Attach the interrupt to count when the signal drops (FALLING edge)
  attachInterrupt(digitalPinToInterrupt(RPM_PIN), countPulse, FALLING);

  // Print the CSV Header so Python knows what the columns are
  Serial.println("Time_ms,Voltage_V,Current_mA,Power_mW,Temp_C,RPM");
}

void loop() {
  // We want to send data EXACTLY once every 1000 milliseconds (1 second)
  if (millis() - lastMillis >= 1000) {
    
    // --- 1. CALCULATE RPM ---
    // Detach interrupt temporarily so the pulseCount doesn't change while we do math
    detachInterrupt(digitalPinToInterrupt(RPM_PIN));
    
    // Standard PC/Laptop fans send 2 pulses per revolution. 
    // pulseCount / 2 = actual revolutions per second.
    // Multiply by 60 to get Revolutions Per Minute (RPM).
    rpm = (pulseCount / 2) * 60; 
    
    // Reset the counter and turn the interrupt back on
    pulseCount = 0;
    attachInterrupt(digitalPinToInterrupt(RPM_PIN), countPulse, FALLING);

    // --- 2. READ POWER (INA219) ---
    float busvoltage = ina219.getBusVoltage_V();
    float current_mA = ina219.getCurrent_mA();
    float power_mW = ina219.getPower_mW();

    // --- 3. READ TEMPERATURE (DS18B20) ---
    tempSensor.requestTemperatures(); // Command the sensor to read
    float temp_C = tempSensor.getTempCByIndex(0);

    // --- 4. PRINT TO SERIAL CABLE ---
    // Format: Time_ms, Voltage, Current, Power, Temp, RPM
    Serial.print(millis());
    Serial.print(",");
    Serial.print(busvoltage, 2); // Print with 2 decimal places
    Serial.print(",");
    Serial.print(current_mA, 2);
    Serial.print(",");
    Serial.print(power_mW, 2);
    Serial.print(",");
    Serial.print(temp_C, 2);
    Serial.print(",");
    Serial.println(rpm); // println adds the hidden "Enter" key at the end of the line

    // Update the timer
    lastMillis = millis();
  }
}