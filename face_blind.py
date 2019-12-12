# USAGE
# python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat
# python3 pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
#from RPi import GPIO
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
EYE_AR_THRESHOLD = 0.25
EYE_AR_MAX_FRAMES = 3
MINIMUM_DISTANCE = 12
FRAMES_COUNTER = 0
DEBUGGING = True


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
    #GPIO.output(BUZZER_PIN, 1)
    print("Buzzer on")
def buzzer_off():
    #GPIO.output(BUZZER_PIN, 0)
    print("Buzzer off")
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
args_parser.add_argument("-w", "--webcam", type=int, default=0,
                         help="index of webcam on system")
args = args_parser.parse_args()

if args.verbosity:
    print('Verbosity turned on')

# 	help="get more info")
# args = vars(ap.parse_args())


# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor

print("[INFO] loading facial landmark predictor...")
#detector = cv2.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# print(detector.__repr__())
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print('[INFO] pin init')
#init_pins()
print("[INFO] starting video stream thread...")


# vs = VideoStream(usePiCamera=True).start()
vs = VideoStream(src = 0).start()
#vs = VideoStream().start()

# time to warm up camera
time.sleep(10.0)
cv2.imshow("frame", vs.read())
time.sleep(10)
print("Detection started")
# loop over frames from the video stream
while True:
    try:
        frame = vs.read()

        print(frame.size)
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
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
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True

                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                        '''
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm,
                                       args=(args["alarm"],))
                            t.deamon = True
                            t.start()
                        '''
                    # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        if DEBUGGING:
            cv2.imshow("Frame", frame)
            if COUNTER > EYE_AR_THRESH:
                print("Eye frames counter: " + str(COUNTER) + " Thereshold is: " + str(EYE_AR_THRESH))

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cleanup()
            break

    except e:
        print("EXCEPTIOn")
        cv2.destroyAllWindows()
        # GPIO.cleanup()
        vs.stop()
        break



































        '''
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = vs.read()
        print(frame.size)
        print(frame.size())
        frame = imutils.resize(frame, width=400)
        print("resized")
        print(frame.size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        #rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          #minNeighbors=5, minSize=(30, 30),
                                          #flags=cv2.CASCADE_SCALE_IMAGE)


        # loop over the face detections
        for (x, y, w, h) in rects:
            # construct a dlib rectangle object from the Haar cascade
            # bounding box left top
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
            if ear < EYE_AR_THRESHOLD:
                FRAMES_COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # frames, then sound the alarm
                if COUNTER >= EYE_AR_MAX_FRAMES:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        #buzzer_on()

                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False
                #buzzer_off()

            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters

            cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if 1 > 0 :
            print("LEGIA")
            cv2.imshow("Frame", frame)
        #blind_spot()
        else:
            pass


    except:
        print("EXCEPTIOn")
        cv2.destroyAllWindows()
    #    #GPIO.cleanup()
        vs.stop()
        break
'''