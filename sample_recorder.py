from pynput import keyboard
import threading
import cv2 as cv
import time
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from main import ball_deteccion, ceil_deteccion, obstacle_deteccion, measure_distance_to_ceil_and_detted_colission, analize_space
import cv2
import statistics as stats
#cv.startWindowThread()
#cv.namedWindow("preview")

folder_dest = "samples"
time_start = 0
kernel_two = np.ones((1,1),np.uint8)
def analize_space(img, ceil_x, ceil_y, delta_y=15, delta_x=110, steep_x=3, canvas_img=None): 
    ceil_all_mask = cv.morphologyEx(img,cv.MORPH_OPEN,kernel_two)
    ceil_all_mask = cv.morphologyEx(ceil_all_mask,cv.MORPH_CLOSE,kernel_two)
    #cv.imshow("2", ceil_all_mask)
    def analize_point(mask, x, y):
        try:
            if mask[y, x] < 100:
                return 1
            elif mask[y, x] == 100:
                return -1
        except:
            pass
        return 0

    
    cv.line(canvas_img, (ceil_x, ceil_y+delta_y), (ceil_x-delta_x, ceil_y-10), (0,0,255))
    cv.line(canvas_img, (ceil_x, ceil_y+delta_y), (ceil_x+delta_x, ceil_y-10), (0,0,255))
    cy_ = ceil_y+delta_y
    last_cy = 0
    last_cx = 0
    found_left = False
    left_distance = 0
    left_list = []
    for cx_ in range(ceil_x, ceil_x-delta_x, -steep_x):
        cy_ -= 0.3*steep_x
        cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),1)
        result = analize_point(ceil_all_mask, cx_, math.floor(cy_))
        left_list.append(result)
        if result == 1:
            #print(cx_, math.floor(cy_))
            #print("empty space left")
            cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,255,0),10)  
            left_distance = (cx_ - ceil_x)*-1
            
        elif result == -1:
            #print(cx_, math.floor(cy_))
            #print("obs detect left")
            cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),10)  
            
    cy_ = ceil_y+delta_y
    found_right = False
    right_list = []
    right_distance = 0
    for cx_ in range(ceil_x, ceil_x+delta_x, steep_x):
        cy_ -= 0.3 * steep_x
        cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),1)
        result = analize_point(ceil_all_mask, cx_, math.floor(cy_))
        right_list.append(result)
        if result == 1:
            #print(cx_, math.floor(cy_))
            #print("empty space right")
            cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,255,0),10)
            right_distance = cx_ - ceil_x
            found_right = True
            
        elif result == -1:
            #print(cx_, math.floor(cy_))
            #print("obs detect right")
            cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),10) 
    
    return [right_list, left_list]

class recorder():
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.__frame = None
        self.__time_start = 0
        self.__lock = threading.Lock()
        self.__press = False
        self._ceil = {'x': [], 'y': []}
        self._data = []
        self.__capture_counter = 0
        self.measure_ceil_distance(100)
        
    def measure_ceil_distance(self, repeat = 100):
        ceil = {'x': [], 'y': []}
        for _ in range(repeat):
            _, img = self.cap.read()
            img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)    
            data = []
            cx, cy, _ = ball_deteccion(img_hsv, img)
            ceil_mask = ceil_deteccion(img_hsv)
            obs_mask = obstacle_deteccion(img_hsv) 

            obs_all_mask = ceil_mask + obs_mask
            ceil_cx, ceil_cy, ceil_distance, colision = measure_distance_to_ceil_and_detted_colission(obs_all_mask, cx, cy)
            if ceil_cx == 0:
                continue
            ceil['x'].append(ceil_cx)
            ceil['y'].append(ceil_cy)
        self._ceil['y'] = int(stats.median(ceil['y']))
        self._ceil['x'] = int(stats.median(ceil['x']))
        print("Ceil position {} {}".format(self._ceil['x'], self._ceil['y']))

    def processing_img(self, img, key):
        self.__lock.acquire()
        print("Capture {}".format(self.__capture_counter))
        self.__capture_counter += 1
        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)    
        data = []
        cx, cy, _ = ball_deteccion(img_hsv, img)
        ceil_mask = ceil_deteccion(img_hsv)
        obs_mask = obstacle_deteccion(img_hsv) 

        obs_all_mask = ceil_mask + obs_mask
        ceil_cx, ceil_cy, ceil_distance, colision = measure_distance_to_ceil_and_detted_colission(obs_all_mask, cx, cy)       

        ceil_space = analize_space(obs_all_mask,  self._ceil['x'],  self._ceil['y'], canvas_img=img, steep_x=1, delta_y=20)
        air_space = analize_space(obs_all_mask,  self._ceil['x'], self._ceil['y']-75, canvas_img=img, steep_x=1, delta_y=0)
        next_ceil_space = analize_space(obs_all_mask,  self._ceil['x'], self._ceil['y']+150, canvas_img=img, steep_x=1, delta_y=25)

        data.append(str(key))
        data.append(ceil_distance)
        data.append(colision)
        data.append(ceil_space)
        data.append(air_space)
        data.append(next_ceil_space)

        self._data.append(data)
        self.__capture_counter += 1
        self.__lock.release()
        #cv.imshow("preview", img)
        cv2.imshow('H', img)
        cv2.waitKey(0)
        


    def processing(self, key):
        _, self.__frame = self.cap.read()
        thread = threading.Thread(target=self.processing_img, args=(self.__frame, key,))
        thread.start()
        
        
        

    def on_press(self, key):   
        if key == keyboard.Key.esc:
            self.save()            
            raise     
        if (key == keyboard.Key.left or key == keyboard.Key.right or key == keyboard.Key.space):
            #self.__lock.acquire()
            self.__press = True
            self.processing(key)
            print("Start capture")
            self.time_start = time.time()         
                      

    def on_release(self, key):
        if key == keyboard.Key.left or key == keyboard.Key.right or key == keyboard.Key.space:
            time_delta = time.time() - self.__time_start
            print("End Capture")
            filename = "{}/{}-{}d-{}s.JPG".format(folder_dest, time.time(), str(key), time_delta)
            thread = threading.Thread(target=self.write, args=(filename, self.__frame,))
            thread.start()
            self.__press = False
            #self.__lock.release()
    
    def write(self, filename, frame):
        cv.imwrite(filename, frame)
        #cv.imwrite()

    def save(self):
        df = pd.DataFrame(self._data,columns=['key','ceil_distance', 'colision', 'ceil_space', 'air_space', 'next_level_space'])
        df.to_csv(str(time.time()) + ".csv")

r = recorder()
# Collect events until released
with keyboard.Listener(
        on_press=r.on_press) as listener:
    listener.join()