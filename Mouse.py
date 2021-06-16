import pyautogui
import win32api, win32con
import time

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

            x1 = mouse_x + diff_x * 20
            y1 = mouse_y + diff_y * 20
            # print(x1, y1)

            self.last_y = y
            self.last_x = x
            self.change_pos(x1, y1)

    def change_mouse_hand(self, x, y):
        diff_x = x - self.last_x_hand
        diff_y = y - self.last_y_hand

        if(diff_x<0  and abs(diff_x)>10 and diff_y in range(-5, 5)):
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

        elif(diff_x>0 and abs(diff_x)>10 and diff_y in range(-5, 5)):
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


        if (diff_y > 0 and abs(diff_y)>10 and diff_x in range(-10, 10)):
            direction = 'UP'
            self.hand_points['UP'].append(diff_y)
            if (len(self.hand_points['UP']) > 3):
                print('UP')
                self.change_pos(pyautogui.position().x, pyautogui.position().y)
                time.sleep(0.01)
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL,0,0,120)
                self.hand_points['UP'].clear()
                self.hand_points['DOWN'].clear()
        elif (diff_y < 0 and abs(diff_y)>10 and diff_x in range(-10, 10)):
            direction = 'DOWN'
            self.hand_points['DOWN'].append(diff_y)
            if (len(self.hand_points['DOWN']) > 3):
                print('DOWN')
                self.change_pos(pyautogui.position().x, pyautogui.position().y)
                time.sleep(0.01)
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL,0,0,-120)
                self.hand_points['DOWN'].clear()
                self.hand_points['UP'].clear()
        self.last_y_hand = y
        self.last_x_hand = x

