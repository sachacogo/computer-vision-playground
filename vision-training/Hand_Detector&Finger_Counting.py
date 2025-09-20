import cv2
import mediapipe as mp 

mp_hands = mp.solutions.hands  # shortcut to access MediaPipe's "hands" solution
mp_drawing = mp.solutions.drawing_utils  # shortcut to access MediaPipe's drawing utilities
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)  
# activate the hand detection AI with confidence thresholds

cap = cv2.VideoCapture(0)  # select camera (0 = default camera)

def count_fingers(hand_landmarks):  # hand_landmarks corresponds to all detected hand points, it's a MediaPipe object
    # Indices of fingertip landmarks in MediaPipe
    tips = [4, 8, 12, 16, 20]
    fingers = []  # state of fingers (1 = open, 0 = closed), associated with the tips list

    # Thumb (different because horizontal)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x: #not an actual finger counting, just analyzing if the thumb's tip is to the left of the base (so only works for right hand)
        # hand_landmarks -> .landmark (select a specific point) -> .x (x-coordinate of the point)
        fingers.append(1)  # thumb is open
    else:
        fingers.append(0)  # thumb is closed

    # Other fingers (vertical)
    for tip in tips[1:]:  # for each fingertip except the thumb
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:  #not an actual finger counting, just analyzing if the finger's tip is above the base
            # compare tip (fingertip) to the joint two indices before (finger base)
            fingers.append(1)  # finger is open
        else:
            fingers.append(0)  # finger is closed
    return sum(fingers)  # return the total number of fingers raised

while True:
    ret, frame = cap.read()  # returns a boolean (True/False) if capture succeeded, and the captured image (frame)
    if not ret:
        break 
    frame = cv2.flip(frame, 1)  # flip the image horizontally for a mirror effect

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    # convert image from BGR (OpenCV default) to RGB (required by MediaPipe)
    result = hands.process(rgb_frame)  
    # process the image with MediaPipe AI to detect hands

    # hand_landmarks is a list of objects representing detected hand landmarks in the image
    # landmark is a list of individual points for a specific hand
    # multi_hand_landmarks is a list of all detected hands in the image

    if result.multi_hand_landmarks:  # if at least one hand is detected
        for hand_landmarks in result.multi_hand_landmarks:  # for each detected hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  
            # draw landmarks and connections on the original image
            total_fingers = count_fingers(hand_landmarks)  # count raised fingers for this hand
            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()
