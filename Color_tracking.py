import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while(1):

    # Take each frame
    _, frame = cap.read()
    
    if(frame == None):
        print("cannot access frame")
        break
    

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # define range of green color in HSV
    lower_green = np.array([50,110,50])
    upper_green = np.array([255,130,255])
    
    # define range of red color in HSV
    lower_red = np.array([50,50,110])
    upper_red = np.array([255,255,130])    


    ## Threshold the HSV image to get only blue colors
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    ## Red
    #mask = cv2.inRange(hsv, lower_red, upper_red)


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()