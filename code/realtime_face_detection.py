# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:34:48 2022

@author: Viet Nguyen
"""

import cv2
import numpy as np
import face_recognition

webcame = cv2.VideoCapture(0)
face_location = []

modi_image = face_recognition.load_image_file('images/modi.jfif')
modi_encoding = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/trump.jpg')
trump_encoding = face_recognition.face_encodings(trump_image)[0]

kim_da_mi_image = face_recognition.load_image_file('images/kim_da_mi.webp')
kim_da_mi_encoding = face_recognition.face_encodings(kim_da_mi_image)[0]



park_seo_joon_image2 = face_recognition.load_image_file('images/park_seo_joon_2.jpg')
park_seo_joon_encoding2 = face_recognition.face_encodings(park_seo_joon_image2)[0]

park_seo_joon_image3 = face_recognition.load_image_file('images/park_seo_joon_3.webp')
park_seo_joon_encoding3 = face_recognition.face_encodings(park_seo_joon_image3)[0]

park_seo_joon_image4 = face_recognition.load_image_file('images/park_seo_joon_4.jpg')
park_seo_joon_encoding4 = face_recognition.face_encodings(park_seo_joon_image4)[0]

known_face_encodings = [modi_encoding, trump_encoding, kim_da_mi_encoding, park_seo_joon_encoding2,park_seo_joon_encoding3,park_seo_joon_encoding4]
image_labels = ["Modi", "Trump", "Kim da mi", "Park Seo Yoon2", "Park Seo Yoon3", "Park Seo Yoon4"]

while True:
    ret, current_frame = webcame.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small, model='hog')
    all_face_encoding = face_recognition.face_encodings(current_frame_small, all_face_locations)
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encoding):
        top_pros, right_pros, bottom_pros, left_pros = current_face_location
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
        faceDis = face_recognition.face_distance(known_face_encodings,current_face_encoding)
        matchIndex = np.argmin(faceDis)
        name_of_person = 'Khong biet'
        top_pros = top_pros * 4
        right_pros = right_pros * 4
        bottom_pros = bottom_pros * 4
        left_pros = left_pros * 4
        if all_matches[matchIndex]:
            name_of_person = image_labels[matchIndex]
        cv2.rectangle(current_frame, (left_pros, top_pros), (right_pros, bottom_pros), (0, 0, 255), 2)  
        print(name_of_person)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pros, bottom_pros), font, 0.5, (255, 255, 255))
        
    cv2.imshow('Video Webcame', current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
       break     
        
webcame.release()
cv2.destroyAllWindows()