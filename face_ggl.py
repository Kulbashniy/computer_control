import cv2
import mediapipe as mp
import pyautogui as pg
import math
import time
from Mouse import Mouse
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

mouse = Mouse()

def euclidean_distance(point1, point2):
    # try:
    #
    # except ValueError:
    #     return -1
    # print(((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2))
    return math.sqrt(((int(point1[0]) - int(point2[0]))**2) + ((int(point1[1]) - int(point2[1]))**2))

DISTANCE = None

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
clicked = 0
with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        count = 0
        for face_landmarks in results.multi_face_landmarks:
            if(DISTANCE == None):
                landmark_387 = face_landmarks.landmark
                landmark_387 = landmark_387[386]
                x = landmark_387.x
                y = landmark_387.y
                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])
                cv2.circle(image, (relative_x, relative_y), radius=5, color=(225, 0, 100), thickness=1)

                landmark_370 = face_landmarks.landmark
                landmark_370 = landmark_370[374]
                x = landmark_370.x
                y = landmark_370.y
                shape = image.shape
                relative_x_370 = int(x * shape[1])
                relative_y_370 = int(y * shape[0])
                cv2.circle(image, (relative_x_370, relative_y_370), radius=5, color=(225, 0, 100), thickness=1)
                dist = euclidean_distance((relative_x_370, relative_y_370), (relative_x, relative_y))
                DISTANCE = dist
                continue
            else:
                landmark_387 = face_landmarks.landmark
                landmark_387 = landmark_387[386]
                x = landmark_387.x
                y = landmark_387.y
                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])
                cv2.circle(image, (relative_x, relative_y), radius=5, color=(225, 0, 100), thickness=1)

                landmark_370 = face_landmarks.landmark
                landmark_370 = landmark_370[374]
                x = landmark_370.x
                y = landmark_370.y
                shape = image.shape
                relative_x_370 = int(x * shape[1])
                relative_y_370 = int(y * shape[0])
                cv2.circle(image, (relative_x_370, relative_y_370), radius=5, color=(225, 0, 100), thickness=1)
                dist = euclidean_distance((relative_x_370, relative_y_370), (relative_x, relative_y))
                diff = DISTANCE - dist
                if(DISTANCE*diff/100 > 0.50):
                    clicked+=1
                    print(clicked)
                    mouse.click(pg.position().x, pg.position().y)
                    time.sleep(0.2)

                landmark_face = face_landmarks.landmark
                landmark_face = landmark_face[9]
                x = landmark_face.x
                y = landmark_face.y
                shape = image.shape
                relative_x_face = int(x * shape[1])
                relative_y_face = int(y * shape[0])
                mouse.change_mouse_face(relative_x_face, relative_y_face)
                cv2.circle(image, (relative_x_face, relative_y_face), radius=5, color=(225, 0, 100), thickness=1)

            # for landmark in face_landmarks.landmark:
            #     x = landmark.x
            #     y = landmark.y
            #
            #     shape = image.shape
            #     relative_x = int(x * shape[1])
            #     relative_y = int(y * shape[0])
            #     count += 1
            #     # cv2.putText(image, str(count), (relative_x, relative_y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255),
            #     #             1)
            #     if(count == 387 or count == 375 ):
            #         cv2.circle(image, (relative_x, relative_y), radius=5, color=(225, 0, 100), thickness=1)
            #         continue



                # cv2.circle(image, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=1)
    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
