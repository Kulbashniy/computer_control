# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:21:18 2020

@author: hp
"""

import cv2
import mediapipe
import numpy as np
import pyautogui
import win32api
import win32con

import math
import time
from threading import Thread

from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
last_x = 0
last_y = 0


class Mouse:
    def __init__(self):
        self.last_x = pyautogui.position().x
        self.last_y = pyautogui.position().y
        self.last_x_hand = 0
        self.last_y_hand = 0
        self.hand_points = {
            'RIGHT':[],
            'LEFT':[],
            'UP':[],
            'DOWN':[],
        }

    def click(self, x, y):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    def change_pos(self, x, y):
        win32api.SetCursorPos((x, y))

    def change_mouse_face(self, x, y, first=False):
        """
        Calculate difference in eye positions and change mouse pos
        :param x: Position of Eye in oX
        :param y: Position of Eye in oY
        :return:
        """


        diff_x = x - self.last_x
        diff_y = y - self.last_y

        if(abs(diff_x)>1 or abs(diff_y)>1):
            mouse_x = pyautogui.position().x
            mouse_y = pyautogui.position().y

            x1 = mouse_x - diff_x * 20
            y1 = mouse_y + diff_y * 20
            # print(x1, y1)

            self.last_y = y
            self.last_x = x
            self.change_pos(x1, y1)

    def change_mouse_hand(self, x, y):
        diff_x = x - self.last_x_hand
        diff_y = y - self.last_y_hand

        if(diff_x<0  and abs(diff_x)>10 and diff_y in range(-20, 20)):
            direction = 'RIGHT'
            self.hand_points['RIGHT'].append(diff_x)
            if(len(self.hand_points['RIGHT'])>10):
                print('RIGHT')
                pyautogui.keyDown('alt')
                time.sleep(.2)
                pyautogui.press('shift')
                time.sleep(.2)
                pyautogui.press('tab')
                time.sleep(.2)
                pyautogui.keyUp('alt')
                pyautogui.keyUp('shift')
                pyautogui.keyUp('tab')

                self.hand_points['RIGHT'].clear()

        elif(diff_x>0 and abs(diff_x)>10 and diff_y in range(-20, 20)):
            direction = 'LEFT'
            self.hand_points['LEFT'].append(diff_x)
            if (len(self.hand_points['LEFT']) > 10):
                print('LEFT')

                pyautogui.keyDown('alt')
                time.sleep(.2)
                pyautogui.press('tab')
                time.sleep(.2)
                pyautogui.keyUp('alt')
                pyautogui.keyUp('tab')

                self.hand_points['LEFT'].clear()


        if (diff_y > 0 and abs(diff_y)>10 and diff_x in range(-20, 20)):
            direction = 'UP'
            self.hand_points['UP'].append(diff_y)
            if (len(self.hand_points['UP']) > 10):
                print('UP')
                pyautogui.click()
                pyautogui.scroll(100)
                self.hand_points['UP'].clear()
        elif (diff_y < 0 and abs(diff_y)>10 and diff_x in range(-20, 20)):
            direction = 'DOWN'
            self.hand_points['DOWN'].append(diff_y)
            if (len(self.hand_points['DOWN']) > 10):
                print('DOWN')
                pyautogui.click()
                pyautogui.scroll(-100)
                self.hand_points['DOWN'].clear()
        self.last_y_hand = y
        self.last_x_hand = x





mouse = Mouse()


def midpoint(point1, point2):
    return (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2


def euclidean_distance(point1, point2):
    # try:
    #
    # except ValueError:
    #     return -1
    # print(((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2))
    return math.sqrt(((int(point1[0]) - int(point2[0]))**2) + ((int(point1[1]) - int(point2[1]))**2))


def get_blink_ratio(eye_points, facial_landmarks):
    # loading all the required points
    corner_left = (facial_landmarks[36][0], facial_landmarks[36][1])
    corner_right = (facial_landmarks[39][0], facial_landmarks[39][1])

    center_top = midpoint(facial_landmarks[37], facial_landmarks[38])
    center_bottom = midpoint(facial_landmarks[41], facial_landmarks[40])

    # calculating distance
    horizontal_length = euclidean_distance(corner_right, corner_left)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also find the extreme points of each eye

    Parameters
    ----------
    mask : np.uint8
        Blank mask to draw eyes on
    side : list of int
        the facial landmark numbers of eyes
    shape : Array of uint32
        Facial landmarks

    Returns
    -------
    mask : np.uint8
        Mask with region of interest drawn
    [l, t, r, b] : list
        left, top, right, and bottommost points of ROI

    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    if cx - end_points[2] and end_points[3] - cy:
        x_ratio = (end_points[0] - cx)/(cx - end_points[2])
        y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img, end_points, right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up

    """
    global mouse
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)



        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            text = 'Looking left'
        elif left == 2:
            print('Looking right')
            text = 'Looking right'
        elif left == 3:
            print('Looking up')
            text = 'Looking up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 255), 2, cv2.LINE_AA)


def hand_ggl():
    drawingModule = mediapipe.solutions.drawing_utils
    handsModule = mediapipe.solutions.hands
    #global cap
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,
                           max_num_hands=1) as hands:
        while (True):

            global ret
            global img

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

            # cv2.imshow('Test hand', img)

            if cv2.waitKey(1) == 27:
                break



face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
  
cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

thrd = Thread(target=hand_ggl)
thrd.start()


cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
# cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

BLINK_RATIO_THRESHOLD = 6

while(True):
    ret, img = cap.read()
    rects = find_faces(img, face_model)
    for rect in rects:

        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        face_point = midpoint((rect[0], rect[1]), (rect[2], rect[3]))
        cv2.circle(img, (int(face_point[0]), int(face_point[1])), 5, (255, 0, 0), -1)
        mouse.change_mouse_face(int(face_point[0]), int(face_point[1]))

        shape = detect_marks(img, landmark_model, rect)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)

        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = 75
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)

        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)

        right_eye_ratio = get_blink_ratio(right, shape)
        blink_ratio = (right_eye_ratio)

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            mouse.click(pyautogui.position().x, pyautogui.position().y)

        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

        cv2.imshow('eyes', img)
        # cv2.imshow("image", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
cap.release()
cv2.destroyAllWindows()
