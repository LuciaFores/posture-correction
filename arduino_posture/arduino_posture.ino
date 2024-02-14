#include <LiquidCrystal.h>

const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

int redPin= 6;
int greenPin = 9;
int bluePin = 10;

void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  lcd.begin(16, 2);
  Serial.begin(9600);
}

void setColor(int redValue, int greenValue, int blueValue) {
  analogWrite(redPin, redValue);
  analogWrite(greenPin, greenValue);
  analogWrite(bluePin, blueValue);
}

void loop() {
  if (Serial.available()) {
      char serialListener = Serial.read();
      if (serialListener == 'R') {
        setColor(255, 0, 0);
        lcd.clear();
        lcd.print("Warning!");
      }
      else if (serialListener == 'G') {
        setColor(0, 255, 0);
        lcd.clear();
        lcd.print("Ok!");
      }
      else if (serialListener == 'Y') {
        setColor(255, 80, 0);
        lcd.clear();
        lcd.print("Not Aligned");
      }
      else if (serialListener == 'V') {
        setColor(128, 0, 128);
        lcd.clear();
        lcd.print("Computing...");
      }
      else if (serialListener == 'O') {
        setColor(0, 0, 0);
        lcd.clear();
      }
  }
}