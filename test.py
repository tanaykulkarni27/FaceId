import cv2 
from FaceRecog import FaceHandler
from PIL import Image as im
import time
x = FaceHandler(cv2.imread('test.jpg'))
x = x.get_face_from_image();
image = im.fromarray(x)
image.show();
image = im.open('test.jpg')
image.show();