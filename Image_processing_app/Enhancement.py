import cv2
import streamlit as st
import  numpy as np

def Histogram_Equaliza(img):
    im=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eq=cv2.equalizeHist(im)
    e=cv2.cvtColor(eq,cv2.COLOR_GRAY2RGB)
    return e

def Bilateral(img):
    blur = cv2.bilateralFilter(img,9,75,75)
    blur=cv2.cvtColor(blur,cv2.COLOR_BGR2RGB)
    return blur

def Laplacian(img):
    kernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    a=cv2.filter2D(img,-1,kernel)
    return a

def Smoothing(img):
    kernel=1/9*(np.array([[1,1,1],[1,1,1],[1,1,1]]))
    s=cv2.filter2D(img,-1,kernel)
    s=cv2.cvtColor(s,cv2.COLOR_BGR2RGB)
    return s

def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)

def apply_contrast_stretching(img):
    # Apply contrast stretching
    r1 = st.slider('r1', 0, 255, 70)
    s1 = st.slider('s1', 0, 255, 0)
    r2 = st.slider('r2', 0, 255, 140)
    s2 = st.slider('s2', 0, 255, 255)
    contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)

    contrast_stretched = np.clip(contrast_stretched, 0, 255)
    contrast_stretched = contrast_stretched.astype(np.uint8)

    contrast_stretched=cv2.cvtColor(contrast_stretched,cv2.COLOR_BGR2RGB)
    return contrast_stretched