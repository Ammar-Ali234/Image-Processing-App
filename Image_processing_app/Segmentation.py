import cv2
import streamlit as st
import  numpy as np


def Thresholding(img):
    t=st.slider("Select Threshold Value",0,255,100)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,th=cv2.threshold(img,t,255,cv2.THRESH_BINARY)
    return th

def Sobel(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    i=cv2.filter2D(img,-1,kernel)
    return i

def Canny(img):
    a=cv2.Canny(img,100,200,None)
    return a
