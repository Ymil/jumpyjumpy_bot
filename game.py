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
from tensorflow.keras import models
#cv.startWindowThread()
#cv.namedWindow("preview")
from pynput.keyboard import Key, Controller
keyboard = Controller()
folder_dest = "samples"
time_start = 0
kernel_two = np.ones((1,1),np.uint8)
def analize_point(mask, x, y):
    try:
        if mask[y, x] < 100:
            return 1
        elif mask[y, x] == 100:
            return -1
    except Exception as e:
        print(e)
    return 0

def analize_space(img, pos_x, pos_y, width=200, canvas_img=None):
    start_x = math.floor(pos_x-(width/2))
    end_x = math.floor(pos_x+(width/2))
    cv.line(canvas_img, (start_x, pos_y), (end_x, pos_y), (0,0,255))
    line_result = []
    for cx_ in range(start_x, end_x):
        cv.circle(canvas_img,(cx_, pos_y),2,(0,0,255),1)
        result = analize_point(img, cx_, pos_y)
        line_result.append(result)
        if result == 1:
            cv.circle(canvas_img,(cx_, pos_y),2,(0,255,0),2)       
        elif result == -1:
            cv.circle(canvas_img,(cx_, pos_y),2,(0,0,255),2)
    return line_result

def analize_ceil(img, ceil_x, ceil_y, second_width=45,
    delta_y=15, delta_x=110, steep_x=3, canvas_img=None): 
    time_start_47 = time.time()
    
    ceil_all_mask = cv.morphologyEx(img,cv.MORPH_OPEN,kernel_two)
    ceil_all_mask = cv.morphologyEx(ceil_all_mask,cv.MORPH_CLOSE,kernel_two)
    #cv.imshow("2", ceil_all_mask)
    print("Processing 47 time {}s".format(time.time() - time_start_47))
    #cv.line(canvas_img, (ceil_x, ceil_y+delta_y), (ceil_x-delta_x, ceil_y-10), (0,0,255))
    #cv.line(canvas_img, (ceil_x, ceil_y+delta_y), (ceil_x+delta_x, ceil_y-10), (0,0,255))
    cy_ = ceil_y+delta_y
    last_cy = 0
    last_cx = 0
    
    
    left_list = []
    time_start_61 = time.time()
    for cx_ in range(ceil_x, ceil_x-delta_x, -steep_x):
        cy_ -= 0.3*steep_x
        cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),1)
        result = analize_point(ceil_all_mask, cx_, math.floor(cy_))
        left_list.append(result)
        #if result == 1:
            #print(cx_, math.floor(cy_))
            #print("empty space left")
        #    cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,255,0),10)  
        #    left_distance = (cx_ - ceil_x)*-1            
        #elif result == -1:
            #print(cx_, math.floor(cy_))
            #print("obs detect left")
        #    cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),10)
        #    pass
    print("Processing 61 time {}s".format(time.time() - time_start_61))
    time_start_78 = time.time()
    cy_ = math.floor(cy_)
    for cy_ in range(cy_, cy_-second_width, -steep_x):
        cx_ += 0.3*steep_x
        cx_tmp = math.floor(cx_)
        cv.circle(canvas_img,(cx_tmp, cy_),2,(0,0,255),1)
        result = analize_point(ceil_all_mask, cx_tmp, math.floor(cy_))
        left_list.append(result)
        #if result == 1:
            #print(cx_, math.floor(cy_))
            #print("empty space left")
        #    cv.circle(canvas_img,(cx_tmp, math.floor(cy_)),2,(0,255,0),3)  
        #    left_distance = (cx_ - ceil_x)*-1            
        #elif result == -1:
        #    pass
            #print(cx_, math.floor(cy_))
            #print("obs detect left")
        #    cv.circle(canvas_img,(cx_tmp, math.floor(cy_)),2,(0,0,255),3)
    print("Processing 78 time {}s".format(time.time() - time_start_78))
    cy_ = ceil_y+delta_y
    
    right_list = []
    
    for cx_ in range(ceil_x, ceil_x+delta_x, steep_x):
        cy_ -= 0.3 * steep_x
        cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),1)
        result = analize_point(ceil_all_mask, cx_, math.floor(cy_))
        right_list.append(result)
        #if result == 1:
            #print(cx_, math.floor(cy_))
            #print("empty space right")
        #    cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,255,0),10)
        #    right_distance = cx_ - ceil_x
        #    found_right = True
            
        #elif result == -1:
            #print(cx_, math.floor(cy_))
            #print("obs detect right")
        #    cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),10)
        #    pass

    cy_ = math.floor(cy_)
    for cy_ in range(cy_, cy_-second_width, -steep_x):
        cx_ -= 0.3*steep_x
        cx_tmp = math.floor(cx_)
        cv.circle(canvas_img,(cx_tmp, cy_),2,(0,0,255),1)
        result = analize_point(ceil_all_mask, cx_tmp, math.floor(cy_))
        right_list.append(result)
        #if result == 1:
            #print(cx_, math.floor(cy_))
            #print("empty space left")
        #    cv.circle(canvas_img,(cx_tmp, math.floor(cy_)),2,(0,255,0),3)  
        #    left_distance = (cx_ - ceil_x)*-1            
        #elif result == -1:
            #print(cx_, math.floor(cy_))
            #print("obs detect left")
        #    cv.circle(canvas_img,(cx_tmp, math.floor(cy_)),2,(0,0,255),3)
        #    pass
    return right_list + left_list

class recorder():
    def __init__(self):
        self.cap = cv.VideoCapture(0) 
        self.__frame = None
        self.__time_start = 0
        self.__lock = threading.Lock()
        self.__press = False
        self._ceil = {'x': [], 'y': []}
        self._data = []
        self.data = {}
        self.death = False
        self.__capture_counter = 0
        self._opcion_move = 0
        self._press_key = ''
        self.measure_ceil_distance(100)
        
        thread = threading.Thread(target=self.move)
        thread.start()
        
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
    
    def ceil_thread(self, mask, img):
        ceil_space = analize_ceil(mask,  self._ceil['x'],  self._ceil['y'], canvas_img=img, steep_x=1, delta_y=20)
        self.data['ceil_space'] = ceil_space
    
    def next_ceil_space_thread(self, mask, img):
        next_ceil_space = analize_ceil(mask,  self._ceil['x'], self._ceil['y']+150, canvas_img=img, steep_x=1, delta_y=25, delta_x=90, second_width=65)
        self.data['next_ceil_space'] = next_ceil_space

    def air_space_thread(self, mask, img):
        air_space = analize_space(mask,  self._ceil['x'], self._ceil['y']-100, canvas_img=img, width=310)
        self.data['air_space'] = air_space

    def processing(self, img):
        
        print("Capture {} len {}".format(self.__capture_counter, len(self._data)))
        self.__capture_counter += 1

        time_start = time.time()
        time_start_176 = time.time()
        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        print("Processing 176 time {}s".format(time.time() - time_start_176))
        time_start_179 = time.time()
        ceil_mask = ceil_deteccion(img_hsv)
        obs_mask = obstacle_deteccion(img_hsv) 
        obs_all_mask = ceil_mask + obs_mask           
        print("Processing 179 time {}s".format(time.time() - time_start_179))
        
        time_start_184 = time.time()
        ceil_thread = threading.Thread(target=self.ceil_thread, args=(obs_all_mask, img,))
        next_ceil_space_thread = threading.Thread(target=self.next_ceil_space_thread, args=(obs_all_mask, img,))
        air_space_thread = threading.Thread(target=self.air_space_thread, args=(obs_all_mask, img,))
        ceil_thread.start()
        next_ceil_space_thread.start()
        air_space_thread.start()

        time_start_193 = time.time()
        ceil_thread.join()
        next_ceil_space_thread.join()
        air_space_thread.join()
        #ones[55:255] = air_space
        #air_space = ones
        print("Processing 193 time {}s".format(time.time() - time_start_193))
        print("Processing 184 time {}s".format(time.time() - time_start_179))
        print("Processing area time {}s".format(time.time() - time_start))
        if not stats.mode(self.data['ceil_space']) == 1 and not stats.mode(self.data['next_ceil_space']) == 1:
            data = np.array([self.data['air_space'], self.data['ceil_space'], self.data['next_ceil_space']])
            self._data.append(data)
            if len(self._data) > 1:
                time_start = time.time()
                self.set_move(np.argmax(np.mean(self.model.predict(np.array(self._data)), 0)))
                print(np.mean(self.model.predict(np.array(self._data)), 0))
                print(self._opcion_move)
                del self._data[0]
                print("Processing predic time {}s".format(time.time() - time_start))       
            self.__capture_counter += 1            
        else:
            self._data = []
            print("Muestra descartada")
        
        return img

    
    def set_move(self, opcion_move):
        self._move_last_change = time.time()
        self._opcion_move = opcion_move

    def move(self):
        self._move_last_change = time.time()
        while not self.death:
            if self._opcion_move == 0:
                if len(self._press_key) > 0:
                    last_change = time.time()
                if self._press_key == 'Key.left':
                    keyboard.release(Key.left)
                elif self._press_key == "Key.right":
                    keyboard.release(Key.right)
                self._press_key = ''

            elif self._opcion_move == 2:
                if self._press_key == '' or self._press_key == 'Key.right':
                    if self._press_key == 'Key.right':
                        keyboard.release(Key.right)
                    keyboard.press(Key.left)
                    self._press_key = "Key.left"
                    last_change = time.time()

            elif self._opcion_move == 1:
                if self._press_key == '' or self._press_key == 'Key.left':
                    if self._press_key == 'Key.left':
                        keyboard.release(Key.left)
                    keyboard.press(Key.right)
                    last_change = time.time()
                    self._press_key = "Key.right"
                        
            if time.time() - self._move_last_change > 0.5:
                self._move_last_change = time.time()
                self._opcion_move = 0
    
    def load_model(self):
        # cargar json y crear el modelo
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = models.model_from_json(loaded_model_json)
        # cargar pesos al nuevo modelo
        self.model.load_weights("model.h5")
        print("Cargado modelo desde disco.")
        
        # Compilar modelo cargado y listo para usar.
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

r = recorder()
r.load_model()
if __name__ == "__main__":    
    while True:
        # Capture frame-by-frame
        start = time.time()
        ret, frame = r.cap.read()
        print("Capture {}s".format(time.time() - start))
        start = time.time()
        img = r.processing(frame)
        print("Processing {}s".format(time.time() - start))
        cv.imshow("1", img)
        
        if cv.waitKey(1) == ord('q'):
            break