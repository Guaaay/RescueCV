import RPi.GPIO as GPIO
import threading
import time

class Beeper:
    def __init__(self, pin):
        self.pin = pin
        self.frequency = 0
        self.pwm = None
        self.thread = None
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)

    def set_frequency(self, frequency):
        self.frequency = frequency

    def start_beeping(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._beep_thread)
            self.thread.start()

    def _beep_thread(self):
        while True:
            if self.frequency > 0:
                if self.pwm is None:
                    self.pwm = GPIO.PWM(self.pin, self.frequency)
                    self.pwm.start(50)
                else:
                    self.pwm.ChangeFrequency(self.frequency)
            else:
                if self.pwm is not None:
                    self.pwm.stop()
                    self.pwm = None
            time.sleep(1)
            
    def pause_beeping(self):
        self.frequency = 0

    def stop_beeping(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
            if self.pwm is not None:
                self.pwm.stop()
                self.pwm = None
                
    def cleanup(self):
        GPIO.cleanup()

# Example usage
if __name__ == "__main__":
    try:
        beeper = Beeper(pin=2)
        beeper.set_frequency(10)  
        beeper.start_beeping()
        time.sleep(2)  
        print("new cycle")
        beeper.set_frequency(1000)
        time.sleep(2)
        print("pausing")
        beeper.pause_beeping()
        time.sleep(2)
        print("resume")
        beeper.set_frequency(5)
        beeper.start_beeping()
        time.sleep(2)
        beeper.pause_beeping()
        beeper.cleanup()
    except Exception as e:
        print(e)
    finally:
        GPIO.cleanup()

