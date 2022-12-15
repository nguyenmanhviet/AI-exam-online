import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from imutils import face_utils
import imutils
import dlib
import os

def e_dist(pA, pB):
	return np.linalg.norm(pA - pB)

def eye_ratio(eye):    
	d_V1 = e_dist(eye[1], eye[5])
	d_V2 = e_dist(eye[2], eye[4])
	d_H = e_dist(eye[0], eye[3])
	eye_ratio_val = (d_V1 + d_V2) / (2.0 * d_H)
	return eye_ratio_val

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

eye_ratio_threshold = 0.2
head_ratio_threshold = 0.25

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(base64.b64decode(image_encoded)))
    return pil_image

def pil_cv(pil_image):
    opencv_image = np.array(pil_image.convert("RGB"))
    opencv_image = opencv_image[:, :, ::-1].copy()
    return opencv_image

def max_face(face):
    w_max = 0
    index = 0
    for i, (x, y, w, h) in enumerate(face):
        if w >= w_max:
            w_max = w
            index = i
    return index

def detect(opencv_image):
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1,	minNeighbors=5, minSize=(100, 100),	flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0: 

        index_f = max_face(faces)

        (x,y,w,h) = faces[index_f]

        # for (x, y, w, h) in faces:

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        cv2.rectangle(opencv_image, (x,y), (x+w,y+h), (255,0,0), 3)

        landmark = landmark_detect(gray, rect)
        landmark = face_utils.shape_to_np(landmark)

        leftEye = landmark[left_eye_start:left_eye_end]
        rightEye = landmark[right_eye_start:right_eye_end]

        nose = landmark[29]
        left_head = landmark[1]
        right_head = landmark[15]

        left_eye_ratio = eye_ratio(leftEye)
        right_eye_ratio = eye_ratio(rightEye)

        eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        left_eye_bound = cv2.convexHull(leftEye)
        right_eye_bound = cv2.convexHull(rightEye)
        cv2.drawContours(opencv_image, [left_eye_bound], -1, (0, 255, 0), 1)
        cv2.drawContours(opencv_image, [right_eye_bound], -1, (0, 255, 0), 1)

        left_head_ratio = e_dist(nose, left_head) / w
        right_head_ratio = e_dist(nose, right_head) / w

        head_avg_ratio = (left_head_ratio + right_head_ratio)/2.0

        cv2.line(opencv_image, left_head, nose, (0,255,255), 2)
        cv2.line(opencv_image, right_head, nose, (255,255,0), 2)

        if eye_avg_ratio < eye_ratio_threshold:
            cv2.putText(opencv_image, "LOW  -  Eye", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return 0
        else:
            if left_head_ratio < head_ratio_threshold or right_head_ratio < head_ratio_threshold:
                cv2.putText(opencv_image, "LOW  -  Head", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                return 0
            else:
                cv2.putText(opencv_image, "EYE AVG RATIO: {:.3f}".format(eye_avg_ratio), ((x+120,y-50)),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(opencv_image, "HEAD AVG RATIO: {:.3f}".format(head_avg_ratio), ((x+120,y-20)),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(opencv_image, "HIGH!!!", (x,y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                return 1

    else: 
#        cv2.putText(opencv_image, "LOW!!!", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return 0

def save_image(opencv_image):
    folder = '.'

    cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(folder,'face.jpg'),opencv_image)
    # image = cv2.cvtColor(cv2.imread('D:\ANSON\DATN\code\\face_image\\face.jpg'),cv2.COLOR_BGR2RGB)
    # cv2.imshow('face',image)


