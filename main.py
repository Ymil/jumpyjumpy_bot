# Autor: Lautaro Linquiman - lylinquiman@gmail.com

import cv2 as cv
import numpy as np
import statistics as stats

from pynput.keyboard import Key, Controller
from queue import Queue

import time
import numpy as np
import math
import threading


# Umbrales de colores
ball_min = (136, 0, 66)
ball_max = (299, 360, 270)
ceil_min = (0, 182, 0)
ceil_max = (107, 360, 360)
obs_min = (50, 84, 202)
obs_max = (142, 229, 265)
from pynput.keyboard import Key, Controller
keyboard = Controller()
kernel = np.ones((5,5),np.uint8)
kernel_two = np.ones((1,1),np.uint8)
tendency = 0
results = []
results_last_time = time.time()
evasion_flag = False
fall_flag = False
last_tendency_change = 0
ceil_position_list = {'x': [], 'y':[]}
q = Queue()

def evasion(q_):
    global evasion_flag       
    print("Evadiendo")
    keyboard.press(Key.left)    
    while q_.get():
        pass
    evasion_flag = False
    time.sleep(0.1)
    keyboard.release(Key.left)
    print("Evadido")



def ball_deteccion(img, canvas_img):
    '''
    return: position ball and mask [x, y, mask]
    '''
    mask = cv.inRange(img, ball_min, ball_max)

    #Eliminando ruido
    
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

#    cv.imshow("", mask)
#    cv.waitKey()
    cx = 0
    cy = 0
    #print((contours, _))
    for i in contours:
        #Calcular el centro a partir de los momentos
        #  continue
        
        (x,y,w,h) = cv.boundingRect(i)
        if w/h > 1.5:
            continue
        #print("W %s H %s: %s" % (w,h,w/h))    
        #cv.rectangle(canvas_img,(x,y),(x+w,y+h),(0,255,0),2)
        momentos = cv.moments(i)
        cx = int(momentos['m10']/momentos['m00'])
        cy = int(momentos['m01']/momentos['m00'])

        #Dibujar el centro
        #cv.circle(canvas_img,(cx, cy), 3, (0,0,255), -1)
        break
    return [cx, cy, mask] 

def ceil_deteccion(img):
     #Ceil Detection

    ceil_mask = cv.inRange(img, ceil_min, ceil_max)
    return ceil_mask

def obstacle_deteccion(img):
    obs_mask = cv.inRange(img, obs_min, obs_max)
    obs_mask[obs_mask > 0] = 100
    return obs_mask

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
            return False

        
        cv.line(canvas_img, (ceil_x, ceil_y+delta_y), (ceil_x-delta_x, ceil_y-10), (0,0,255))
        cv.line(canvas_img, (ceil_x, ceil_y+delta_y), (ceil_x+delta_x, ceil_y-10), (0,0,255))
        cy_ = ceil_y+delta_y
        found_left = False
        left_distance = 0
        left_list = []
        for cx_ in range(ceil_x, ceil_x-delta_x, -steep_x):
            cy_ -= 0.3*steep_x
            cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),1)
            result = analize_point(ceil_all_mask, cx_, math.floor(cy_))
            if result == 1:
                #print(cx_, math.floor(cy_))
                #print("empty space left")
                cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,255,0),10)  
                left_distance = (cx_ - ceil_x)*-1
                found_left = True
                break
            elif result == -1:
                #print(cx_, math.floor(cy_))
                #print("obs detect left")
                cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),10)  
                return -1            
            
        cy_ = ceil_y+delta_y
        found_right = False
        right_distance = 0
        for cx_ in range(ceil_x, ceil_x+delta_x, steep_x):
            cy_ -= 0.3 * steep_x
            cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),1)
            result = analize_point(ceil_all_mask, cx_, math.floor(cy_))
            if result == 1:
                #print(cx_, math.floor(cy_))
                #print("empty space right")
                cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,255,0),10)
                right_distance = cx_ - ceil_x
                found_right = True
                break
            elif result == -1:
                #print(cx_, math.floor(cy_))
                #print("obs detect right")
                cv.circle(canvas_img,(cx_, math.floor(cy_)),2,(0,0,255),10)  
                return -2
        if found_left and found_right:
            #print(left_distance)
            #print(right_distance)
            if left_distance > right_distance:
                return 1
            elif left_distance < right_distance:
                return 2
        elif found_left:
            return 1
        elif found_right:
            return 2
        return 0

def no_move():    
    keyboard.release(Key.left)        

def move_left(stime=0.03):
    keyboard.press(Key.left)
    time.sleep(stime)
    keyboard.release(Key.left)

def move_right(stime=0.03):    
    keyboard.press(Key.right)
    time.sleep(stime)
    keyboard.release(Key.right)
    
    
def measure_distance_to_ceil_and_detted_colission(mask, position_x, position_y):
    result_cx = 0
    result_cy = 0
    distance = 0
    colision = False
    delta_position_y = 480-position_y
    #print(delta_position_y)
    for y in range(position_y, position_y+1000):
        try:
            if mask[y, position_x] > 0:                
                distance = y-position_y
                result_cx = position_x
                result_cy = y
                if mask[y, position_x] == 100:
                    colision = True
                #print("Distancia suelo px: %s y: %s, value: %s, colision %s" % (distance, y, mask[y, position_x] , colision))
                break
        except Exception as e:
            #print("Error midiendo distancia")
            #print(e)
            distance = 1000
            #print("error")
    return (result_cx, result_cy, distance, colision)
   
def processing(img):
    global last_tendency_change
    global results
    global results_last_time
    global fall_flag
    global tendency
    global ceil_position_list
    global evasion_flag

    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)    
    cx, cy, _ = ball_deteccion(img_hsv, img)
    ceil_mask = ceil_deteccion(img_hsv)
    obs_mask = obstacle_deteccion(img_hsv) 

    obs_all_mask = ceil_mask + obs_mask
    ceil_cx, ceil_cy, ceil_distance, colision = measure_distance_to_ceil_and_detted_colission(obs_all_mask, cx, cy)
    
    if evasion_flag:
        q.put(colision)
        return

    if colision:
        print("Colision detección d: %s" % ceil_distance)
        if ceil_distance < 190:
            evasion_flag = True
            x = threading.Thread(target=evasion, args=(q,))
            x.start()
            return

    if len(ceil_position_list['y']) < 10:
        ceil_position_list['y'].append(ceil_cy)
        ceil_position_list['x'].append(ceil_cx)
    ceil_position_y_prom = int(stats.median(ceil_position_list['y']))
    ceil_position_x_prom = int(stats.median(ceil_position_list['x']))
    
    #print(ceil_distance)

    
    
    if ceil_distance < 100:           
        time_start = time.time()     
        result = analize_space(obs_all_mask, ceil_position_x_prom, ceil_position_y_prom, canvas_img=img)
        #print("Analize space in %s" % (time.time()-time_start))
        results.append(result)
        if not len(results) >= 5:
            return        
        try:
            result = stats.mode(results)
            print(results)
            print("Cantidad de datos alcanzados (%d) %ss result: %d" % (len(results),time.time() - results_last_time, result))
            results_last_time = time.time()
        except Exception as e:
            print(e, results)
            return
        results = []             
        
            
        if tendency == 0:
            if result == 0 or result == 1:
                print("Cambiando tendency 1")
                tendency = 1
            elif result == 2:
                print("Cambiando tendency 2")
                tendency = 2

        if result < 0:
            if result == -1:
                print("OBS Cambiando tendency 2 %s" % result)
                tendency = 2
            elif result == -2:
                print("OBS Cambiando tendency 1 %s" % result)
                tendency = 1
        
        if tendency == 1:
            move_right(0.1)
        elif tendency == 2:
            move_left(0.1)
        
    else:
        no_move()
        results = []
        fall_flag = True
        
    cv.circle(img,(ceil_position_x_prom, ceil_cy),2,(0,255,0),-1)
    cv.line(img,(cx,cy),(ceil_position_x_prom,ceil_cy),(255,0,0),1)

    #numpy_horizontal = np.hstack((img, ceil_mask, obs_mask))
    cv.imshow("1", img)
    cv.imshow("2", ceil_mask)
    cv.imshow("3", obs_mask)
    cv.imshow("4", obs_all_mask)


        
if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        processing(frame)

        if cv.waitKey(1) == ord('q'):
            break