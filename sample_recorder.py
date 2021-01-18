from pynput import keyboard
import threading
import cv2 as cv
import time
from .main import ceil_deteccion, obstacle_deteccion

folder_dest = "/samples"
time_start = 0
class recorder():
    def __init__():
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        self.__frame = None
        self.__time_start = 0
        self.__lock = threading.Lock()

    def processing(self):
        _, __frame = self.cap.read()
        #img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    def on_press(self, key):        
        if key == keyboard.Key.left or key == keyboard.Key.right:
            self.__lock.acquire()
            self.processing()
            print("Start capture")
            self.time_start = time.time()         
                      

    def on_release(self, key):
        if key == keyboard.Key.left or key == keyboard.Key.right:
            time_delta = time.time() - self.__time_start
            print("End Capture")
            filename = "{}/{}-{}d-{}s.JPG".format(folder_dest, time.time(), str(key), time_delta)
            thread = threading.Thread(target=self.write, args=(filename, self.__frame))
            thread.start()
            self.__lock.release()
    
    def write(self, filename, frame):
        cv.imwrite(__frame, filename)

r = recorder()
# Collect events until released
with keyboard.Listener(
        on_press=r.on_press,
        on_release=r.on_release) as listener:
    listener.join()