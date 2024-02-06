import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

def predict_digit(img):
    # Loading the model
    model = tf.keras.models.load_model(
        "NewDigitRecModelv20.h5"
    )

    # Predicting the digit
    img = img.reshape((1, 28, 28))
    img = img / 255.0
    prediction = np.argmax(model.predict([img]))
    return prediction

def extract_digit_contours(image, resolution_factor):
    # Preprocessing Stage
    image_copy = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13,2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the area based on the image resolution
    image_area = image.shape[0] * image.shape[1]

    # Define a factor to adjust the minimum contour area based on image resolution
    resolution_factor = resolution_factor 
    # Calculate the adjusted minimum contour area
    min_contour_area = int(image_area * (resolution_factor / 100))
    digit_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    return digit_contours

def mark_digits(image, digit_contours):
    for cnt in digit_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(image_copy, (x, y), ((x+w), (y+h)), (9, 78, 145), 5)

        if h < w:
            if h > (w * (5/100)):
                cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 0, 0), 5)

        else:
            if w > (h * (5/100)):
                cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 0, 0), 5)

    return image

def mark_and_predict_digits(image, digit_contours):
    # Copying the image so that we can use one for prediction & another one for marking
    image_copy = image.copy()

    font_scale = round(image.shape[0] / 400, 1) # Getting the font scale factor by dividing 400 with height
    print(font_scale)
    for cnt in digit_contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h < w:
            if h > (w * (5/100)):
                cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 0, 0), 5)
                digit = image_copy[y:y + h, x:x + w]
                digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
                _, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                print(digit.shape)
                digit = make_square(digit)
                digit = np.array(Image.fromarray(digit).resize((28, 28)))
                digit = cv2.GaussianBlur(digit, (3, 3), 0)

                predicted_value = predict_digit(digit)

                cv2.putText(
                    image, str(predicted_value),
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    2
                )
        else:
            if w > (h * (5/100)):
                cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 0, 0), 5)
                digit = image_copy[y:y + h, x:x + w]
                digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
                _, digit = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                print(digit.shape)
                digit = make_square(digit)
                digit = np.array(Image.fromarray(digit).resize((28, 28)))
                digit = cv2.GaussianBlur(digit, (3, 3), 0)

                predicted_value = predict_digit(digit)

                cv2.putText(
                    image, str(predicted_value),
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 0, 0),
                    2
                )
    return image

def make_square(image):
    # Get the original image dimensions
    col_channels = None
    
    if len(image.shape) < 3:
        height, width = image.shape
    else:
        height , width, col_channels = image.shape
    
    border = height // 6
    print(border)
 
    big = height
    if big < width: big = width
    # Determine the size of the square (use the height as the size)
    size = big + 2 * border
    # Finding the mean pixel values of the image
    # avg_pixel = int(np.mean(image))
    
    # Creating array based on color channels of the original image
    if col_channels:
        # Create a square canvas
        square_canvas = np.zeros((size, size, col_channels), dtype=np.uint8) # + avg_pixel
    else:
        square_canvas = np.zeros((size, size), dtype=np.uint8) # + avg_pixel

 
    # Calculate the padding (if needed) to center the original image
    pad_x = (size - width) // 2
    pad_y = (size - height) // 2
    # Place the original image on the square canvas
    square_canvas[pad_y: pad_y + height, pad_x:pad_x + width] = image
 
    return square_canvas


