# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 22:03:41 2022

@author: Viet Nguyen
"""

import os
import time
import cv2

number_images = 30


cap = cv2.VideoCapture(0)
def get_image(label):
    idx = 0
    for imgnum in range(number_images):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        name_label = label + '_' + str(idx)
        imgname = os.path.join('./images',f'{name_label}.jpg')
        idx = idx + 1
        print(imgname)
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.5)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

get_image('nga')

cap.release()
cv2.destroyAllWindows()