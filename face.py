#!/usr/bin/python3

import cv2

#cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)

from picamera2 import Picamera2

# Grab images as numpy arrays and leave everything else to OpenCV.

face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
#cv2.startWindowThread()
picam2 = Picamera2()
ww = 640
hh = 480
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (ww, hh)}))
picam2.start()

while True:
    im = picam2.capture_array()

    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in faces:
        #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
        nx=(x+w/2)/ww
        ny=(x+h/2)/hh
        print(f'Face: {nx},{ny}')
    #print("")
    #cv2.imshow("Cam", im)
    # Add a delay to allow the window to refresh
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
