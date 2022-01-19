import cv2
from PIL import Image
class FaceHandler:
    def __init__(self,img):
        self.img = img
        self.CASCADE_PATH = "C:\\Users\\Tanay\\AppData\\Roaming\\Python\\Python310\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml"
    def get_face_from_image(self):
        img = self.img
        face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        x = Image.fromarray(gray_image)
        cnt = 0;
        face = None
        faces = face_cascade.detectMultiScale(gray_image,1.3,1) # return face dimensions 
        for x,y,w,h in faces: 
            if cnt >= 1:
                assert cnt <= 0,"Image should contain single image"
            face_part = gray_image[y : y+h,x:x + w] # slices the faces 
            face = face_part
            cnt += 1
        return face
# get_face_from_image(None)