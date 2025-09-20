import cv2
import numpy as np

cap = cv2.VideoCapture(0) # choose the camera (0 = default camera)

if not cap.isOpened():
    print("error")
    exit()

while True:
    ret, frame = cap.read() # returns a boolean (YES/NO) depending on whether the capture worked and the captured image (frame)
    if not ret:
        print("error")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert the original image from BGR to HSV (choose image and color space)
    # HSV is better suited for color detection as it separates hue and brightness

    low = np.array([90, 50, 50]) # lower bound of the color range to detect
    up = np.array([140, 255, 255]) # upper bound of the color range to detect

    filter = cv2.inRange(hsv, low, up) # filter that keeps only the pixels in the chosen color range
    # (hsv gives the color space, low and up define the color range), output is 255 if in range, 0 otherwise

    result = cv2.bitwise_and(frame, frame, mask=filter) # apply the mask (filter) to the original image (frame)
    # (if filter = 255 -> keep pixel from frame, if filter = 0 -> set pixel from frame to 0)

    cv2.imshow("frame", frame)
    cv2.imshow("filter", filter)
    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'): #if 'q' is pressed,
        break #end of the while loop, exit the program

cap.release() # release the camera for other applications
cv2.destroyAllWindows() # close the windows opened by imshow
