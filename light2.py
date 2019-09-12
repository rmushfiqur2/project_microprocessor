import RPi.GPIO as GPIO
from time import sleep
Servopin = 7
GPIO.setmode(GPIO.BOARD)
GPIO.setup(Servopin, GPIO.OUT)
GPIO.output(Servopin, False)
GPIO.cleanup()
