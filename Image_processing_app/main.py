import os
import easyocr
import cv2
import streamlit as st
from PIL import Image
import numpy as np
import processing
import Enhancement
import Segmentation
import Features

def get_saved_images():
    image_files = [f for f in os.listdir("images") if os.path.isfile(os.path.join("images", f))]
    return image_files

def save_image(img):
    if not os.path.exists("images"):
        os.makedirs("images")
    
    image_num = 1
    while os.path.exists(f"images/img{image_num}.jpg"):
        image_num += 1

    image_name = f"images/img{image_num}.jpg"
    cv2.imwrite(image_name, img)
    return image_name

def create_download_link(image, filename, text):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img.save(filename)
    with open(filename, "rb") as file:
        btn = st.download_button(label=text, data=file, file_name=filename, mime="image/jpeg")
    return btn

def main():
    st.title("Image Processing App")

    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox(
        "Select an option:",
        ["Image Acquisition", "Pre-Processing", "Enhancement", "Segmentations", "Feature Extraction"]
    )

    if menu == "Image Acquisition":
        st.subheader("You have to Upload the Image for starting the processing on it...!")

        up = st.file_uploader("Upload the image to process", type=['jpg', 'jpeg', 'png'])
        if up is not None:
            file = np.asarray(bytearray(up.read()), dtype=np.uint8)
            img = cv2.imdecode(file, 1)
            imgz = cv2.resize(img, (340, 180))  # Resize the image to smaller dimensions

            st.image(imgz, channels='BGR', use_column_width=True)

            if st.button("Proceed"):
                image_name = save_image(img)
                st.success(f"Image saved as {image_name}")
        else:
            st.error("You haven't added the image Yet")

    elif menu == "Pre-Processing":
        st.subheader("Pre-Processing Techniques")
        
        saved_images = get_saved_images()
        selected_image = st.selectbox("Select an image from the saved images:", saved_images)
        
        if selected_image:
            image_path = os.path.join("images", selected_image)
            img = cv2.imread(image_path)
            imgz = cv2.resize(img, (340, 180))  # Resize the image to smaller dimensions

            st.image(imgz, channels='BGR', use_column_width=True)

            pre_process_option = st.sidebar.radio(
                "Select a pre-processing technique:",
                ["Convert to GreyScale", "Median Filtering", "Gaussian Filtering", "Resize the Image"]
            )

            if pre_process_option == "Convert to GreyScale":
                img_grey = processing.Con_to_grey(imgz)
                st.image(img_grey, use_column_width=True)
                create_download_link(img_grey, "Output_Images\\converted_to_greyscale.jpg", "Download GreyScale Image")
            elif pre_process_option == "Median Filtering":
                img_mf = processing.Median_Filter(imgz)
                st.image(img_mf, use_column_width=True)
                create_download_link(img_mf, "Output_Images\\median_filtered.jpg", "Download Median Filtered Image")
            elif pre_process_option == "Gaussian Filtering":
                img_gf = processing.Gaussian_Filter(imgz)
                st.image(img_gf, use_column_width=True)
                create_download_link(img_gf, "Output_Images\\gaussian_filtered.jpg", "Download Gaussian Filtered Image")
            elif pre_process_option == "Resize the Image":
                img_rs = processing.resize_image(imgz)
                st.image(img_rs, use_column_width=True)
                create_download_link(img_rs, "Output_Images\\resized_image.jpg", "Download Resized Image")

    elif menu == "Enhancement":
        st.subheader("Enhancement Techniques")
        saved_images = get_saved_images()
        selected_image = st.selectbox("Select an image from the saved images:", saved_images)
        
        if selected_image:
            image_path = os.path.join("images", selected_image)
            img = cv2.imread(image_path)
            imgz = cv2.resize(img, (340, 180))  # Resize the image to smaller dimensions

            st.image(imgz, channels='BGR', use_column_width=True)

        enhancement_option = st.sidebar.radio("Select an enhancement technique:", ["Histogram Equalization", "Bilateral Filtering", "Laplacian Filtering", "Smoothing", "Contrast Stretching"])
        if enhancement_option == "Histogram Equalization":
            img_eq = Enhancement.Histogram_Equaliza(imgz)
            st.image(img_eq, use_column_width=True)
            create_download_link(img_eq, "Output_Images\\histogram_equalized.jpg", "Download Histogram Equalized Image")
        elif enhancement_option == "Bilateral Filtering":
            img_bl = Enhancement.Bilateral(imgz)
            st.image(img_bl, use_column_width=True)
            create_download_link(img_bl, "Output_Images\\bilateral_filtered.jpg", "Download Bilateral Filtered Image")
        elif enhancement_option == "Laplacian Filtering":
            img_lp = Enhancement.Laplacian(imgz)
            st.image(img_lp, use_column_width=True)
            create_download_link(img_lp, "Output_Images\\laplacian_filtered.jpg", "Download Laplacian Filtered Image")
        elif enhancement_option == "Smoothing":
            img_s = Enhancement.Smoothing(imgz)
            st.image(img_s, use_column_width=True)
            create_download_link(img_s, "Output_Images\\smoothed.jpg", "Download Smoothed Image")
        elif enhancement_option == "Contrast Stretching":
            img_cs = Enhancement.apply_contrast_stretching(imgz)
            st.image(img_cs, use_column_width=True)
            create_download_link(img_cs, "Output_Images\\contrast_stretched.jpg", "Download Contrast Stretched Image")

    elif menu == "Segmentations":
        st.header("Segmentation Techniques")
        saved_images = get_saved_images()
        selected_image = st.selectbox("Select an image from the saved images:", saved_images)
        
        if selected_image:
            image_path = os.path.join("images", selected_image)
            img = cv2.imread(image_path)
            imgz = cv2.resize(img, (340, 180))  # Resize the image to smaller dimensions

            st.image(imgz, channels='BGR', use_column_width=True)

            seg_option = st.sidebar.radio("Select a pre-processing technique:", ["Threshold", "Sobel Filter", "Canny Filter"])
            if seg_option == "Threshold":
                img_th = Segmentation.Thresholding(imgz)
                st.image(img_th, use_column_width=True)
                create_download_link(img_th, "Output_Images\\thresholded.jpg", "Download Thresholded Image")
            elif seg_option == "Sobel Filter":
                img_sob = Segmentation.Sobel(imgz)
                st.image(img_sob, use_column_width=True)
                create_download_link(img_sob, "Output_Images\\sobel_filtered.jpg", "Download Sobel Filtered Image")
            elif seg_option == "Canny Filter":
                img_can = Segmentation.Canny(imgz)
                st.image(img_can, use_column_width=True)
                create_download_link(img_can, "Output_Images\\canny_filtered.jpg", "Download Canny Filtered Image")

    elif menu == "Feature Extraction":
        st.write("Feature Extraction selected. You can extract features from images in this section.")
        st.header("Segmentation Techniques")
        saved_images = get_saved_images()
        selected_image = st.selectbox("Select an image from the saved images:", saved_images)
        if selected_image:
            image_path = os.path.join("images", selected_image)
            img = cv2.imread(image_path)
            imgz = cv2.resize(img, (340, 180))  # Resize the image to smaller dimensions

            st.image(imgz, channels='BGR', use_column_width=True)

            fex_option = st.sidebar.radio("Select a Feature Extraction technique:", ["Mean", "Standard Deviation","Text Extractor"])
            if fex_option == "Mean":
                img_me = Features.Mean(imgz)
                st.write(f"The mean pixel value of the image is: {img_me}")
            elif fex_option == "Standard Deviation":
                img_std = Features.St_deviation(imgz)
                st.write(f"The standard deviation of pixel value of the image is: {img_std}")
            elif fex_option == "Text Extractor":
                img_ex = Features.extract_text(imgz)
                st.write("Extracted Text:")
                for item in img_ex:
                    st.write(item[1])

if __name__ == "__main__":
    main()