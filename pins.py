import json
from

with open("pins.json", "r") as conf_file:
	data = conf_file.read()
gpio = json.loads(data)

# print(gpio.values())

FRONT_ASSIST_STATE = gpio["buttons"]["front_assist"]
BLID_SPOT_STATE = gpio["buttons"]["blind_spot"]
DROW_DET_STATE = gpio["buttons"]["drowssines_detection"]

print(FRONT_ASSIST_STATE)
print(BLID_SPOT_STATE)
print(DROW_DET_STATE)
