import os
import cv2 as cv
import numpy as np
#from main import ceil_deteccion, ball_deteccion, obstacle_deteccion
import time
import pandas as pd
import re
dataset = {
    'name': [],
    'direccion': [],
    'duration': []
}


if __name__ == "__main__":
    folder_samples = 'samples_process'
    training_data = []
    for filename in os.listdir(folder_samples):
        if filename[0] == ".":
            continue
        full_filename = folder_samples+"/"+filename
        
        m = re.search('(.*).JPG', filename)
        filename_out_extension = m.group(1)
        m = re.search('.*Key.(.*)d.*.JPG', filename)
        direccion = m.group(1)
        m = re.search('.*-(\d*.\d*)s.JPG', filename)
        duration = m.group(1)

        img = cv.imread(full_filename, cv.COLOR_BGR2RGB)
        training_data.append(img)
        img=np.array(img)
        img = img.astype('float32')
        img /= 255 
        print(img.shape)

        dataset['name'].append(filename_out_extension)
        dataset['direccion'].append(direccion)
        dataset['duration'].append(duration)

    filename = 'samples_dataset/{}'.format(time.time())
    np.save(filename, np.array(training_data))
    df = pd.DataFrame(dataset, columns= ['name', 'direccion', 'duration'])
    df.to_csv(filename+'.csv')