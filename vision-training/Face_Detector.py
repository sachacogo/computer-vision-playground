import cv2
import mediapipe as mp

mp_facemesh = mp.solutions.face_mesh #shortcut to access MediaPipe's "face mesh" solution
mp_drawing = mp.solutions.drawing_utils #shortcut to access MediaPipe's drawing utilities
facemesh = mp_facemesh.FaceMesh(min_detection_confidence = 0.5, max_num_faces=2) #activate the face detection AI with confidence threshold and max number of faces to detect

cap = cv2.VideoCapture(0)

while True:
    bool, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = facemesh.process(rgb_frame) #process the image with MediaPipe AI to detect faces

    if result.multi_face_landmarks : #if at least one face is detected
        for face_landmark in result.multi_face_landmarks: #for each detected face
            mp_drawing.draw_landmarks(frame, face_landmark) #draw landmarks and connections on the original image   
    
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
