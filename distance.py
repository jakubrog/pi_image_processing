#import libs
import RPi.GPIO as GPIO
import time
#BCM numbering scheme
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
#Set pins
LED = 40
TRIG=16
ECHO=12
MINIMUM_DISTANCE = 50

D_T = 0.2
#Set direction
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(LED, GPIO.OUT)


#Endless loop
def distance():
    #Pulse to start measure with HC-SR04
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

    return distance

while True:
  s0 = distance()
  time.sleep(D_T)
  s1 = distance()
  v = (s0-s1)/D_T
  a = 5
  sh = v/(2*a)
  tr = 1
  if sh + 1 * v > s1:
    GPIO.output(LED, 1)
  else:
    GPIO.output(LED, 0)

  # print(v)
  # time.sleep(2)
