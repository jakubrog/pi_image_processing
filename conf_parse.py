import json

with open("conf.json", "r") as conf_file:
    data = conf_file.read()
    x = json.loads(data)


print(x['front_assist']['acceleration'])
