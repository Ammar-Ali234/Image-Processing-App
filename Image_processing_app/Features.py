import cv2
import numpy as np
import streamlit as st 
import easyocr

def Mean(img):
    mean=np.mean(img)
    return mean

def St_deviation(img):
    std_dev_value = np.std(img)
    return std_dev_value

def extract_text(image):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image)
    return result