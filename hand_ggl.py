import cv2
import mediapipe
from Mouse import Mouse

mouse = Mouse()


def hand_ggl():
    drawingModule = mediapipe.solutions.drawing_utils
    handsModule = mediapipe.solutions.hands
    cap = cv2.VideoCapture(0)
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                           max_num_hands=2) as hands:
        while (True):
            ret, img = cap.read()
            thresh = img.copy()

            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    for point in handsModule.HandLandmark:
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                                  normalizedLandmark.y,
                                                                                                  frameWidth,
                                                                                                  frameHeight)

                        cv2.circle(img, pixelCoordinatesLandmark, 5, (0, 255, 0), -1)
                        try:
                            mouse.change_mouse_hand(pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1])
                        except TypeError:
                            pass
                        break

            cv2.imshow('Test hand', img)

            if cv2.waitKey(1) == 27:
                break
hand_ggl()

