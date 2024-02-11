#include <LiquidCrystal.h>

const int rs = 12, en = 11, d4 = 5, d5 = 4, d6 = 3, d7 = 2;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

void setup(){
  lcd.begin(16, 2);
  Serial.begin(9600);
}

void loop(){
  if (Serial.available()){
    char serialListener = Serial.read();
    if (serialListener == 'R') {
        lcd.clear();
        lcd.print("Warning!");
      }
      else if (serialListener == 'G') {
        lcd.clear();
        lcd.print("Ok!");
      }
      else if (serialListener == 'V') {
        lcd.clear();
        lcd.print("Not Aligned");
      }
      else if (serialListener == 'O') {
        lcd.clear();
        lcd.print("Off");
      }
  }
}