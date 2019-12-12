# USAGE
# python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat
# python3 pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
from RPi import GPIO
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import json

# TODO:
# 	dodac debuggowanie, if DEBUGOWANIE: show some info, display video
#	naprawa front assist
#	refaktoring nazw
# 	komentarze
# drow det przycisk cos nie tak 


# configuration
with open("conf.json", "r") as conf_file:
	data = conf_file.read()
values = json.loads(data)

with open("pins.json", "r") as conf_file:
	data = conf_file.read()
gpio = json.loads(data)


EYE_AR_THRESH = values["drowssines_detection"]["ratio"]
CLOSED_EYES_ALARM_FRAMES = values["drowssines_detection"]["closed_eyes_alarm_frames"]
CLOSED_EYES_LED_FRAMES = values["drowssines_detection"]["closed_eyes_led_frames"]
MINIMUM_DISTANCE = values["blind_spot"]["minimum_distance"]
DEBBUGING = values["configuration"]["debbuging"]
ALARM_TIME = values["configuration"]["alarm_time"]
ACCELERATION = values["front_assist"]["acceleration"]
REACTION_TIME = values["front_assist"]["reaction_time"]
BUTTON_LIMIT = values["configuration"]["pushed_button_time"]

COUNTER = 0
ALARM_ON = False
LED_ON = False

FRONT_TRIG = gpio["front_assist"]["trigger"]
FRONT_ECHO = gpio["front_assist"]["echo"]

FRONT_ASSIST_STATE = gpio["buttons"]["front_assist"]
BLID_SPOT_STATE = gpio["buttons"]["blind_spot"]
DROW_DET_STATE = gpio["buttons"]["drowssines_detection"]
FRONT_ASSIST_ENABLE = True
BLIND_SPORT_ENABLE = True
DROW_DET_ENABLE = True
BUTTON_COUNTER = 0


def blind_spot():
	dist = distance(gpio["blind_spot"]["trigger"], gpio["blind_spot"]["echo"])
	if(dist < MINIMUM_DISTANCE):
		GPIO.output(gpio["blind_spot"]["led"], True)
	else:
		GPIO.output(gpio["blind_spot"]["led"], False)


def distance(trigger, echo):
	GPIO.output(trigger, True)
	time.sleep(0.00001)
	GPIO.output(trigger, False)
	#Wait for HIGH on ECHO
	while GPIO.input(echo) == 0:
		pulse_start = time.time()

	#wait for LOW again
	while GPIO.input(echo) == 1:
		pulse_end = time.time()
		signalDelay = pulse_end - pulse_start

	#divider for uS to  s
	constDivider = 1000000/58
	distance = int(signalDelay * constDivider)
	return distance


def init_pins():
	GPIO.cleanup()
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
	GPIO.setup(gpio["buttons"]["blind_spot"], GPIO.IN)
	GPIO.output(gpio["blind_spot"]["enable"], BLIND_SPORT_ENABLE)
	GPIO.output(gpio["front_assist"]["enable"], FRONT_ASSIST_ENABLE)
	GPIO.output(gpio["drowssines_detection"]["enable"], DROW_DET_ENABLE)


def buzzer_on():
	GPIO.output(gpio["sound"]["buzzer"], 1)

def buzzer_off():
	GPIO.output(gpio["sound"]["buzzer"], 0)

# compute and return the euclidean distance between the two points
def euclidean_dist(pointA, pointB):
	return np.linalg.norm(pointA - pointB)

# compute the euclidean distances between the two sets of
# vertical eye landmarks (x, y)-coordinates
def eye_aspect_ratio(eye):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])

	# compute the euclidean distance between the horizontal eye landmark (x, y)
	C = euclidean_dist(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def read_state():
	global BUTTON_COUNTER
	global FRONT_ASSIST_ENABLE
	global BLIND_SPORT_ENABLE
	global DROW_DET_ENABLE
	print(str(GPIO.input(FRONT_ASSIST_STATE)) + str(GPIO.input(BLID_SPOT_STATE)) + str(GPIO.input(DROW_DET_STATE)))

	if not GPIO.input(FRONT_ASSIST_STATE):
		BUTTON_COUNTER += 1
		if BUTTON_COUNTER > BUTTON_LIMIT:
			BUTTON_COUNTER = 0
			GPIO.output(gpio["front_assist"]["enable"], not FRONT_ASSIST_ENABLE)
			FRONT_ASSIST_ENABLE = not FRONT_ASSIST_ENABLE

	if not GPIO.input(BLID_SPOT_STATE):
		BUTTON_COUNTER += 1
		if BUTTON_COUNTER > BUTTON_LIMIT:
			BUTTON_COUNTER = 0
			GPIO.output(gpio["blind_spot"]["enable"], not BLIND_SPORT_ENABLE)
			BLIND_SPORT_ENABLE = not BLIND_SPORT_ENABLE

	if not GPIO.input(DROW_DET_STATE):
		BUTTON_COUNTER += 1
		if BUTTON_COUNTER > BUTTON_LIMIT:
			BUTTON_COUNTER = 0
			GPIO.output(gpio["drowssines_detection"]["enable"], not DROW_DET_ENABLE)
			DROW_DET_ENABLE = not DROW_DET_ENABLE


if DEBBUGING:
	print('[INFO] Debugging turned on')

# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor

vs = VideoStream(src=0).start()
# vs = VideoStream(src = 0, usePiCamera=True).start()

print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print('[INFO] pin init')

init_pins()

if DEBBUGING:
	print("[INFO] starting video stream thread...")


# time to warm up camera
# time.sleep(10.0)
print("[INFO] Detection started")
# loop over frames from the video stream
while True:
	starting_distance = distance(FRONT_TRIG, FRONT_ECHO)
	starting_time = time.time()
	read_state()

	# noinspection PyBroadException
	try:
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)

		frame = vs.read()
		if DROW_DET_ENABLE:
			frame = imutils.resize(frame, width=400)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				# detect faces in the grayscale frame
			rects = detector.detectMultiScale(gray, scaleFactor=1.1,
			minNeighbors=5, minSize=(30, 30),
				flags=cv2.CASCADE_SCALE_IMAGE)

			# loop over the face detections
			for (x, y, w, h) in rects:
				# construct a dlib rectangle object from the Haar cascade
				# bounding box
				rect = dlib.rectangle(int(x), int(y), int(x + w),
					int(y + h))

				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)

				# average the eye aspect ratio together for both eyes
				ear = (leftEAR + rightEAR) / 2.0

				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes

				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)

				# check to see if the eye aspect ratio is below the blink
				# threshold, and if so, increment the blink frame counter
				if ear < EYE_AR_THRESH:
					COUNTER += 1

					# if the eyes were closed for a sufficient number of
					# frames, then sound the alarm
					if COUNTER >= CLOSED_EYES_LED_FRAMES:
						if not LED_ON:
							GPIO.output(gpio["drowssines_detection"]["led"], True)


					if COUNTER >= CLOSED_EYES_ALARM_FRAMES:
						# if the alarm is not on, turn it on
						if not ALARM_ON:
							ALARM_ON = True
							buzzer_on()



						# draw an alarm on the frame
						# cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
						# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				# otherwise, the eye aspect ratio is not below the blink
				# threshold, so reset the counter and alarm
				else:
					COUNTER = 0
					ALARM_ON = False
					LED_ON = False
					buzzer_off()
					GPIO.output(gpio["drowssines_detection"]["led"], False)

			# draw the computed eye aspect ratio on the frame to help
			# with debugging and setting the correct eye aspect ratio
			# thresholds and frame counters
			if DEBBUGING:
				cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if FRONT_ASSIST_ENABLE:
			current_time = time.time()
			current_distance = distance(FRONT_TRIG, FRONT_ECHO)

			current_speed = (starting_distance - current_distance) / (current_time - starting_time)

			breaking_distance = current_speed / (2 * ACCELERATION)

			if (breaking_distance + REACTION_TIME * current_speed) > current_distance:
				GPIO.output(gpio["front_assist"]["led"], 1)
			else:
				GPIO.output(gpio["front_assist"]["led"], 0)

		if BLIND_SPORT_ENABLE:
			blind_spot()

	except:
		cv2.destroyAllWindows()
		GPIO.cleanup()
		vs.stop()
		break
