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

# configuration
FRONT_ASSIST_ENABLE = False
BLIND_SPOT_ENABLE = False
DROW_DET_ENABLE = True
BUTTON_COUNTER = 0

COUNTER = 0
ALARM_ON = False
LED_ON = False


# blind spot monitor
def blind_spot():
	dist = pins.read_distance(pins.BLIND_SPOT_TRIGGER, pins.BLIND_SPOT_ECHO)
	if dist < values.MINIMUM_DISTANCE:
		pins.blind_spot(True)
	else:
		pins.blind_spot(False)

# read distance from sensor
def distance(trigger, echo):
	pins.init_distance_sensor(trigger)
	pulse_start = 0
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

	# compute distance between the vertical eye landmark
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])

	# compute distance between the horizontal eye landmark (x, y)
	C = euclidean_dist(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# read state of buttons and enable or disable modules
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
	print('[WARNING] debugging turned on')

# start videostream
vs = VideoStream(src=0).start()


print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print('[INFO] pins initialization')

init_pins()

print("[INFO] detection started")

while True:

	if FRONT_ASSIST_ENABLE:
		starting_distance = pins.read_distance(pins.FRONT_TRIG, pins.FRONT_ECHO)
		starting_time = time.time()

	read_state()

	# noinspection PyBroadException
	try:

		frame = vs.read()
		if DROW_DET_ENABLE:
			frame = imutils.resize(frame, width=400)
			# make frame grayscale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# detect faces in the grayscale frame
			rects = detector.detectMultiScale(gray, scaleFactor=1.1,
			minNeighbors=5, minSize=(30, 30),
				flags=cv2.CASCADE_SCALE_IMAGE)

			# loop over the face detection
			for (x, y, w, h) in rects:
				# construct a dlib rectangle object from the Haar cascade
				# bounding box
				rect = dlib.rectangle(int(x), int(y), int(x + w),
					int(y + h))

				# determine the facial landmarks for the face region
				shape = predictor(gray, rect)
				# convert coordinates to a NumPy array
				shape = face_utils.shape_to_np(shape)

				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)

				# average of the eye aspect ratio
				ear = (leftEAR + rightEAR) / 2.0


				# if debugging is on compute the convex hull for both eyes,
				# then visualize each of them
				if values.DEBBUGING:
					leftEyeHull = cv2.convexHull(leftEye)
					rightEyeHull = cv2.convexHull(rightEye)
					cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				# check if eyes are closed
				if ear < values.EYE_AR_THRESH:
					COUNTER += 1

					# if the eyes were closed for a sufficient number of
					# frames, then turn on led
					if COUNTER >= values.CLOSED_EYES_LED_FRAMES:
						if not LED_ON:
							pins.drowsiness_detection(True)

					if COUNTER >= values.CLOSED_EYES_ALARM_FRAMES:
						if not ALARM_ON:
							ALARM_ON = True
							pins.buzzer_on()

						# draw an alarm on the frame
						if values.DEBBUGING:
							cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				# if eyes are open reset
				else:
					COUNTER = 0
					ALARM_ON = False
					LED_ON = False
					pins.buzzer_off()
					pins.drowsiness_detection(False)


		if FRONT_ASSIST_ENABLE:
			current_time = time.time()
			# read current distance from a vehicle ahead
			current_distance = pins.read_distance(pins.FRONT_TRIG, pins.FRONT_ECHO)
			# compute current speed of the vehicle
			current_speed = (starting_distance - current_distance) / (current_time - starting_time)
			# compute avaerage breaking distance
			breaking_distance = current_speed * current_speed / (2 * values.ACCELERATION)

			# if collision is possible, sound the alarm
			if (breaking_distance + values.REACTION_TIME * current_speed) > current_distance:
				pins.front_assist(True)
			else:
				pins.front_assist(False)

		# check if blind spot is clear
		if BLIND_SPOT_ENABLE:
			blind_spot()

		# if Debbuging show image on screen
		if values.DEBBUGING:
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

	except:
		cv2.destroyAllWindows()
		pins.gpio_cleanup()
		vs.stop()
		break
