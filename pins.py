import json
import time
from RPi import GPIO


with open("pins.json", "r") as conf_file:
	data = conf_file.read()
gpio = json.loads(data)


FRONT_TRIG = gpio["front_assist"]["trigger"]
FRONT_ECHO = gpio["front_assist"]["echo"]

FRONT_ASSIST_STATE = gpio["buttons"]["front_assist"]
BLIND_SPOT_STATE = gpio["buttons"]["blind_spot"]
DROW_DET_STATE = gpio["buttons"]["drowssines_detection"]


BLIND_SPOT_TRIGGER = gpio["blind_spot"]["trigger"]
BLIND_SPOT_ECHO = gpio["blind_spot"]["echo"]
BLIND_SPOT_LED = gpio["blind_spot"]["led"]
DROWSINESS_DETECTION_PIN = gpio["drowssines_detection"]["led"]


def buzzer_on():
	GPIO.output(gpio["sound"]["buzzer"], 1)

def buzzer_off():
	GPIO.output(gpio["sound"]["buzzer"], 0)

def enable_blind_spot(value):
	GPIO.output(gpio["blind_spot"]["enable"], value)

def enable_front_assist(value):
	GPIO.output(gpio["front_assist"]["enable"], value)

def enable_blind_spot(value):
	GPIO.output(gpio["blind_spot"]["enable"], value)

def enable_drosiness_detection(value):
	GPIO.output(gpio["drowssines_detection"]["enable"], value)

def init():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setwarnings(False)

	for section, value in gpio.items():
		for name, pin in value.items():
			if name == "echo" or section == "buttons":
				if section == "buttons":
					GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
				else:
					GPIO.setup(pin, GPIO.IN)
			else:
				GPIO.setup(pin, GPIO.OUT)

def front_assist_button():
	return not GPIO.input(FRONT_ASSIST_STATE)

def blind_spot_button():
	return not GPIO.input(BLIND_SPOT_STATE)

def drowsiness_detection_button():
	return not GPIO.input(DROW_DET_STATE)

def drowsiness_detection(alarm):
	GPIO.output(gpio["drowssines_detection"]["led"], alarm)

def front_assist(value):
	GPIO.output(gpio["front_assist"]["led"], value)


def blind_spot(value):
	GPIO.output(BLIND_SPOT_LED, value)

def gpio_cleanup():
	GPIO.cleanup()

def init_distance_sensor(trigger):
	GPIO.output(trigger, True)
	time.sleep(0.00001)
	GPIO.output(trigger, False)

def read_state(echo):
	return GPIO.input(echo)

def read_distance(trigger, echo):
    GPIO.output(trigger, True)
    time.sleep(0.00001)
    GPIO.output(trigger, False)

    while not read_state(echo):
        pulse_start = time.time()

    signalDelay = 0
    while read_state(echo):
        pulse_end = time.time()
        signalDelay = pulse_end - pulse_start

    constDivider = 1000000/58
    distance = int(signalDelay * constDivider)
    return distance
