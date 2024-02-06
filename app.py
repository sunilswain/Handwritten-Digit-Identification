import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import *



def main():
    st.title("Handwritten Digit Identification App")
    st.caption("Try uploading an image")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    # Add a slider for selecting the resolution factor
    resolution_factor = st.slider(
        "Choose the appropriate resolution factor (0.01 to 0.5)",
        0.0,
        0.5,
        step=0.01,
        value=0.03
    )
    st.caption("The resolution factor basically helps to identify digit images & crop them")
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)

        # Convert the image to NumPy array
        image = np.array(image)

        # Get digit contours
        digit_contours = extract_digit_contours(image, resolution_factor)
        # Extract the digits 
        digit_marked_image = mark_digits(image.copy(), digit_contours)

        # Display the blurred image
        st.image(digit_marked_image, caption=f"minimum digit area: ({int(resolution_factor * (image.shape[0] * image.shape[1]))})",  channels="RGB", width=500)

        # Button for predicting
        if st.button('Predict'):

            # Display the blurred image
            predicted_marked_image = mark_and_predict_digits(image, digit_contours)
            st.write('Done ✅')

            st.image(predicted_marked_image, caption=f"minimum digit area: ({int(resolution_factor * (image.shape[0] * image.shape[1]))})",  channels="RGB", width=500)

if __name__ == "__main__":
    main()
