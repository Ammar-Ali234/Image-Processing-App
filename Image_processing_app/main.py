import cv2
import streamlit as st
from PIL import Image
import numpy as np

def Con_to_grey(img):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return grey


def Edge(img):
    edge=cv2.Canny(img,100,200)
    return edge

def Sobel(img):
    sob = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    return sob


def ImgBlur(img):
    blur=cv2.GaussianBlur(img,(5,5),0)
    return blur

def face(img):
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fac = fc.detectMultiScale(gr, 1.1, 4)
    for (x, y, w, h) in fac:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return img

# a=cv2.imread("D:\Programming\Image_processing_app\me.jpg")
# G=Sobel(a)
# cv2.imshow("Image",G)
# cv2.waitKey(5000)

st.title("Image Processing App")
st.subheader("You have to Upload the Image for starting the processing on it...!")

up=st.file_uploader("Upload the image to process",type=['jpg','jpeg','png'])
if up is not None:
    file=np.asarray(bytearray(up.read()),dtype=np.uint8)
    img=cv2.imdecode(file,1)
    imgz=cv2.resize(img,(480,640))

    st.image(imgz,channels='BGR',use_column_width=True)

    if st.button("Convert to GreyScale"):   
        img_grey=Con_to_grey(imgz)
        st.image(img_grey,use_column_width=True)

else:
    st.error("You haven't added the image Yet")
      