import RPi.GPIO as GPIO
import time
import subprocess

# Define button pins
BUTTON1 = 17
BUTTON2 = 27
BUTTON3 = 22
RESET_BUTTON = 23  # New reset button

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON3, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(RESET_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Global variable to store running process
current_process = None

def stop_process():
    global current_process
    if current_process is not None:
        current_process.terminate()
        current_process.wait()
        current_process = None

def button_pressed(channel):
    global current_process
    stop_process()  # Stop hoja any running script before starting a new one

    if channel == BUTTON1:
        current_process = subprocess.Popen(["python", "FinalTrain.py"])
    elif channel == BUTTON2:
        current_process = subprocess.Popen(["python", "handProximity.py"])
    elif channel == BUTTON3:
        current_process = subprocess.Popen(["python", "objectDetection.py"])
    elif channel == RESET_BUTTON:
        stop_process()  # Just stop without starting a new process

# Add event detection
GPIO.add_event_detect(BUTTON1, GPIO.FALLING, callback=button_pressed, bouncetime=300)
GPIO.add_event_detect(BUTTON2, GPIO.FALLING, callback=button_pressed, bouncetime=300)
GPIO.add_event_detect(BUTTON3, GPIO.FALLING, callback=button_pressed, bouncetime=300)
GPIO.add_event_detect(RESET_BUTTON, GPIO.FALLING, callback=button_pressed, bouncetime=300)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_process()
    GPIO.cleanup()
