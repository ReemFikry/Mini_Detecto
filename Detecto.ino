#include <Servo.h>

Servo rightServo; // Servo for the right arm
Servo leftServo;  // Servo for the left arm

void setup() {
  rightServo.attach(9); // Attach right servo to pin 9
  leftServo.attach(10); // Attach left servo to pin 10
  rightServo.write(90); // Set initial position to neutral
  leftServo.write(90);
  Serial.begin(9600);   // Begin serial communication
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();
    
    if (command == 'R') {
      rightServo.write(0); // Raise right arm
      delay(1000);         // Hold position
      rightServo.write(90); // Return to neutral
    } 
    else if (command == 'L') {
      leftServo.write(0); // Raise left arm
      delay(1000);        // Hold position
      leftServo.write(90); // Return to neutral
    }
  }
}
