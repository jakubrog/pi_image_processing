import json

with open("conf.json", "r") as conf_file:
	data = conf_file.read()
values = json.loads(data)

EYE_AR_THRESH = values["drowssines_detection"]["ratio"]
CLOSED_EYES_ALARM_FRAMES = values["drowssines_detection"]["closed_eyes_alarm_frames"]
CLOSED_EYES_LED_FRAMES = values["drowssines_detection"]["closed_eyes_led_frames"]
MINIMUM_DISTANCE = values["blind_spot"]["minimum_distance"]
DEBBUGING = values["configuration"]["debbuging"]
ALARM_TIME = values["configuration"]["alarm_time"]
ACCELERATION = values["front_assist"]["acceleration"]
REACTION_TIME = values["front_assist"]["reaction_time"]
BUTTON_LIMIT = values["configuration"]["pushed_button_time"]

