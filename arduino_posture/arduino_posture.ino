int redPin= 9;
int greenPin = 10;
int bluePin = 11;

void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
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
      }
      else if (serialListener == 'G') {
        setColor(0, 255, 0);
      }
      else if (serialListener == 'O') {
        setColor(0, 0, 0);
      }
  }
}