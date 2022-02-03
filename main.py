import cv2 as cv2
import tensorflow as tf
import numpy as np

# detect object in image
# language : python
def object_in_image(image):
    # read image
    img = cv2.imread(image)
    # convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3)
    # draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

if __name__ == '__main__':
    object_in_image('./images/image.jpg')
    