import cv2
import streamlit as st
import numpy as np

def Con_to_grey(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey

def Edge(img):
    edge = cv2.Canny(img, 100, 200)
    return edge

def Sobel(img):
    sob = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    sob = cv2.normalize(sob, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return sob

def ImgBlur(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return blur

def face(img):
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fac = fc.detectMultiScale(gr, 1.1, 4)
    for (x, y, w, h) in fac:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img

st.set_page_config(layout="wide", page_title="Image Processing App", page_icon=":camera:", 
                   initial_sidebar_state="expanded")

st.title("Image Processing App")

st.sidebar.title("Image Processing App")
st.sidebar.write(f"Welcome!")
st.sidebar.write(f"Lets get Started!")
st.sidebar.write("This app allows you to upload an image and perform various image processing operations on it.")

st.subheader("Upload an image to process")

with st.spinner("Loading..."):
    up = st.file_uploader("Upload the image to process", type=['jpg', 'jpeg', 'png'])

if up is not None:
    file = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img = cv2.imdecode(file, 1)
    imgz = cv2.resize(img, (480, 640))

    st.image(imgz, channels='BGR', use_column_width=True)

    col1, col2 = st.columns(2)

    if col1.button("Convert to GreyScale"):
        with st.spinner("Converting to GreyScale..."):
            img_grey = Con_to_grey(imgz)
        col1.image(img_grey, use_column_width=True)

    if col2.button("Detect Edges"):
        with st.spinner("Detecting Edges..."):
            img_edge = Edge(imgz)
        col2.image(img_edge, use_column_width=True)

    col3, col4 = st.columns(2)

    if col3.button("Apply Sobel Filter"):
        with st.spinner("Applying Sobel Filter..."):
            img_sobel = Sobel(imgz)
        col3.image(img_sobel, use_column_width=True)

    if col4.button("Apply Gaussian Blur"):
        with st.spinner("Applying Gaussian Blur..."):
            img_blur = ImgBlur(imgz)
        col4.image(img_blur, use_column_width=True)

    if st.button("Detect Faces"):
        with st.spinner("Detecting Faces..."):
            img_face = face(imgz)
        st.image(img_face, use_column_width=True)
else:
    st.error("You haven't added the image yet")