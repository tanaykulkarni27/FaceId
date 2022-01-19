import cv2 
from FaceRecog import FaceHandler
from PIL import Image as im
import time
x = FaceHandler(cv2.imread('test.jpg'))
x = x.get_face_from_image();
face_cropped_image = im.fromarray(x)
face_cropped_image.show();
original_image = im.open('test.jpg')
original_image.show();