import cv2
from pathlib import Path
import os 
class FaceHandler:
    def __init__(self,img):
        self.img = img
        self.MODULE_DIR = Path(__file__).resolve().parent
        self.CASCADE_DIR = os.path.join(self.MODULE_DIR,'cascades/data/haarcascade_frontalface_alt2.xml')
    def get_face_from_image(self):
        img = self.img
        face_cascade = cv2.CascadeClassifier(self.CASCADE_DIR)
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cnt = 0;
        face = None
        faces = face_cascade.detectMultiScale(gray_image,1.3,1) # returns list of faces
        for x,y,w,h in faces: 
            if cnt >= 1:
                assert cnt <= 0,"Image should contain single face"
            face_part = gray_image[y : y+h,x:x + w] # slices the faces from images
            face = face_part
            cnt += 1
        return face