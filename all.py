# USAGE
# python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat
# python3 pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import json
import pins
import values

# TODO:
# 	dodac debuggowanie, if DEBUGOWANIE: show some info, display video
#	naprawa front assist
#	refaktoring nazw
# 	komentarze
# drow det przycisk cos nie tak 

# configuration

FRONT_ASSIST_ENABLE = True
BLIND_SPOT_ENABLE = True
DROW_DET_ENABLE = True
BUTTON_COUNTER = 0

COUNTER = 0
ALARM_ON = False
LED_ON = False



def blind_spot():
	dist = distance(pins.BLIND_SPOT_TRIGGER, pins.BLIND_SPOT_ECHO)
	if dist < values.MINIMUM_DISTANCE:
		pins.blind_spot(True)
	else:
		pins.blind_spot(False)


def distance(trigger, echo):


	pins.init_distance_sensor(trigger)

	while pins.read_state(echo) == 0:
		pulse_start = time.time()

	#wait for LOW again
	while pins.read_state(echo) == 1:
		pulse_end = time.time()
		signalDelay = pulse_end - pulse_start

	#divider for uS to  s
	constDivider = 1000000/58
	distance = int(signalDelay * constDivider)
	return distance


def init_pins():
	pins.init()
	pins.enable_blind_spot(BLIND_SPOT_ENABLE)
	pins.enable_front_assist(FRONT_ASSIST_ENABLE)
	pins.enable_drosiness_detection(DROW_DET_ENABLE)


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
	global BLIND_SPOT_ENABLE
	global DROW_DET_ENABLE

	if pins.front_assist_button():
		BUTTON_COUNTER += 1
		if BUTTON_COUNTER > values.BUTTON_LIMIT:
			BUTTON_COUNTER = 0
			pins.enable_front_assist(not FRONT_ASSIST_ENABLE)
			FRONT_ASSIST_ENABLE = not FRONT_ASSIST_ENABLE

	if pins.blind_spot_button():
		BUTTON_COUNTER += 1
		if BUTTON_COUNTER > values.BUTTON_LIMIT:
			BUTTON_COUNTER = 0
			pins.enable_blind_spot(not BLIND_SPOT_ENABLE)
			BLIND_SPOT_ENABLE = not BLIND_SPOT_ENABLE

	if pins.drowsiness_detection_button():
		BUTTON_COUNTER += 1
		if BUTTON_COUNTER > values.BUTTON_LIMIT:
			BUTTON_COUNTER = 0
			pins.enable_drosiness_detection(not DROW_DET_ENABLE)
			DROW_DET_ENABLE = not DROW_DET_ENABLE


if values.DEBBUGING:
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

if values.DEBBUGING:
	print("[INFO] starting video stream thread...")


# time to warm up camera
# time.sleep(10.0)
print("[INFO] Detection started")
# loop over frames from the video stream
while True:
	starting_distance = distance(pins.FRONT_TRIG, pins.FRONT_ECHO)
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
				if ear < values.EYE_AR_THRESH:
					COUNTER += 1

					# if the eyes were closed for a sufficient number of
					# frames, then sound the alarm
					if COUNTER >= values.CLOSED_EYES_LED_FRAMES:
						if not LED_ON:
							pins.drowsiness_detection(True)


					if COUNTER >= values.CLOSED_EYES_ALARM_FRAMES:
						# if the alarm is not on, turn it on
						if not ALARM_ON:
							ALARM_ON = True
							pins.buzzer_on()



						# draw an alarm on the frame
						# cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
						# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				# otherwise, the eye aspect ratio is not below the blink
				# threshold, so reset the counter and alarm
				else:
					COUNTER = 0
					ALARM_ON = False
					LED_ON = False
					pins.buzzer_off()
					pins.drowsiness_detection(False)

			# draw the computed eye aspect ratio on the frame to help
			# with debugging and setting the correct eye aspect ratio
			# thresholds and frame counters
			if values.DEBBUGING:
				cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if FRONT_ASSIST_ENABLE:
			current_time = time.time()
			current_distance = distance(pins.FRONT_TRIG, pins.FRONT_ECHO)

			current_speed = (starting_distance - current_distance) / (current_time - starting_time)

			breaking_distance = current_speed / (2 * values.ACCELERATION)

			if (breaking_distance + values.REACTION_TIME * current_speed) > current_distance:
				pins.front_assist(1)
			else:
				pins.front_assist(0)

		if BLIND_SPOT_ENABLE:
			blind_spot()

	except:
		cv2.destroyAllWindows()
		pins.gpio_cleanup()
		vs.stop()
		break
