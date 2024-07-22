import cv2
import streamlit as st

def Con_to_grey(img):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return grey

def Median_Filter(img):
    mf=cv2.medianBlur(img,5)
    mf=cv2.cvtColor(mf,cv2.COLOR_BGR2RGB)
    return mf

def Gaussian_Filter(img):
    gf = cv2.GaussianBlur(img,(5,5),0)
    gf=cv2.cvtColor(gf,cv2.COLOR_BGR2RGB)
    return gf

def resize_image(im):
    width = st.slider("Select width:", 50, 800, 300)
    height = st.slider("Select height:", 50, 800, 300)
    resize_im = cv2.resize(im,(width, height))
    resize=cv2.cvtColor(resize_im,cv2.COLOR_BGR2RGB)
    return resize