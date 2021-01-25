import os
import cv2 as cv
from main import ceil_deteccion, ball_deteccion, obstacle_deteccion


if __name__ == "__main__":
    folder_samples = 'samples'
    output_folder = 'samples_process'
    resize = (64, 48)

    for filename in os.listdir(folder_samples):
        if filename[0] == ".":
            continue
        full_filename = "{}/{}".format(folder_samples, filename)
        img = cv.imread(full_filename, cv.COLOR_RGB2HSV)
        
        ceil = ceil_deteccion(img)
        ceil[ceil > 0] = 255
        obs = obstacle_deteccion(img)
        ball = ball_deteccion(img, None)[2]
        ball[ball > 0] = 50

        all_ = ceil + obs + ball
        print(all_.shape)
        img_end = cv.resize(all_, resize)
        print(img_end.shape)
    

        filename_end = "{}/{}".format(output_folder, filename)
        cv.imwrite(filename_end, img_end)


    