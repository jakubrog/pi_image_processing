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

# pins numbers
TRIG=16
ECHO=12
LED = 40
BUZZER_PIN = 38

# configuration
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
MINIMUM_DISTANCE = 12
COUNTER = 0


ALARM_ON = False

def blind_spot():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    #Wait for HIGH on ECHO
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    #wait for LOW again
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        signalDelay = pulse_end - pulse_start

    #divider for uS to  s
    constDivider = 1000000/58
    distance = int(signalDelay * constDivider)

    if(distance < MINIMUM_DISTANCE):
        GPIO.output(LED, True)
    else:
        GPIO.output(LED, False)



def init_pins():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setwarnings(False)
	GPIO.setup(BUZZER_PIN, GPIO.OUT)
	GPIO.setup(TRIG, GPIO.OUT)
	GPIO.setup(ECHO, GPIO.IN)
	GPIO.setup(LED, GPIO.OUT)

def buzzer_on():
	GPIO.output(BUZZER_PIN, 1)

def buzzer_off():
	GPIO.output(BUZZER_PIN, 0)

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

# construct the argument parse and parse the arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument("-v", "--verbosity", help = "Increase output verbosity")
args = args_parser.parse_args()

if args.verbosity:
	print('Verbosity turned on')

# 	help="get more info")
# args = vars(ap.parse_args())


# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor

vs = VideoStream(src=0).start()
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
print("[INFO] starting video stream thread...")


# vs = VideoStream(usePiCamera=True).start()

# time to warm up camera
# time.sleep(10.0)
print("Detection started")
# loop over frames from the video stream
while True:
    try:
    	# grab the frame from the threaded video file stream, resize
    	# it, and convert it to grayscale
    	# channels)
    	frame = vs.read()
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
    			if COUNTER >= EYE_AR_CONSEC_FRAMES:
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
    			buzzer_off()

    		# draw the computed eye aspect ratio on the frame to help
    		# with debugging and setting the correct eye aspect ratio
    		# thresholds and frame counters

    		# cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
    		# cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    	blind_spot()
    except:
        cv2.destroyAllWindows()
        GPIO.cleanup()
        vs.stop()
        break
