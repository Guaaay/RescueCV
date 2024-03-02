import RPi.GPIO as GPIO
import time

# Set the pin numbering mode
GPIO.setmode(GPIO.BCM)

# Set up the GPIO pin
beeper_pin = 2 # Pin 02 (GPIO 2)
GPIO.setup(beeper_pin, GPIO.OUT)

def turn_on_beeper(pin):
    GPIO.output(pin, GPIO.HIGH)

def turn_off_beeper(pin):
    GPIO.output(pin, GPIO.LOW)

try:
    while True:
        # Turn the beeper on
        turn_on_beeper(beeper_pin)
        print("Beeper ON")
        time.sleep(1)  # Wait for 1 second
        
        # Turn the beeper off
        turn_off_beeper(beeper_pin)
        print("Beeper OFF")
        time.sleep(1)  # Wait for 1 second

except KeyboardInterrupt:
    # Clean up GPIO on keyboard interrupt
    GPIO.cleanup()



