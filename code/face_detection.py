# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:13:32 2022

@author: Viet Nguyen
"""

import os
import cv2
import numpy as np
import face_recognition
import boto3
from PIL import Image
from io import BytesIO

class S3Images(object):
    
    """Useage:
    
        images = S3Images(aws_access_key_id='fjrn4uun-my-access-key-589gnmrn90', 
                          aws_secret_access_key='4f4nvu5tvnd-my-secret-access-key-rjfjnubu34un4tu4', 
                          region_name='eu-west-1')
        im = images.from_s3('my-example-bucket-9933668', 'pythonlogo.png')
        im
        images.to_s3(im, 'my-example-bucket-9933668', 'pythonlogo2.png')
    """
    
    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name):
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, 
                                     aws_secret_access_key=aws_secret_access_key, 
                                     region_name=region_name)
        self.all_file = self.s3.list_objects(Bucket='exam-online-pdf')['Contents']
            # print(key['Key'])
        

    def from_s3(self, bucket, key):
        file_byte_string = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        return Image.open(BytesIO(file_byte_string))
    
    def read_all_from_s3(self):
        return self.all_file

webcame = cv2.VideoCapture(0)

images = S3Images(aws_access_key_id='AKIA6PXUR5ATAWHI2CC4', 
                          aws_secret_access_key='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', 
                          region_name='ap-southeast-2')


all_file = images.read_all_from_s3()

for file in all_file:
    #img = images.from_s3('exam-online-pdf', file)
    # print('file', file)
    image = images.from_s3('exam-online-pdf', file['Key'])
    print(image)
      #  pix = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
      #  print('im', im)
    name = './images/' + file['Key']
    image.save(name)

known_face_encodings = []
image_labels = []


known_face_encodings = []
image_labels = []


path = r'./images'
for filename in os.listdir(path):
    file = './images/' + str(filename)
    print('file', type (file))
    face_image = face_recognition.load_image_file(file)
    print('face_image', type (face_image))
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)  
    label = filename.split('.')[0].split('_')[0]
    print(label)
    image_labels.append(label)
    
#all_file = images.read_all_from_s3()

#for file in all_file:
#    img = images.from_s3('exam-online-pdf', file)
#    face_encoding = face_recognition.face_encodings(img)[0]
#    known_face_encodings.append(face_encoding)
#    label = file.split('.')[0].split('_')[0]
#    print(label)
#    image_labels.append(label)
    
    
while True:
    ret, current_frame = webcame.read()
    print()
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
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pros, bottom_pros), font, 0.5, (255, 255, 255))
        
    cv2.imshow('Video Webcame', current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
       break     
        
webcame.release()
cv2.destroyAllWindows()    


