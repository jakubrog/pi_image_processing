import json

with open("pins.json", "r") as conf_file:
	data = conf_file.read()
gpio = json.loads(data)

# print(gpio.values())

for z, p in gpio.items():
	if z != "buttons":
		for x, y in p.items():
			if x == "echo":
				print("in")
			else:
				print("out")
			print(x)
			print(y)
