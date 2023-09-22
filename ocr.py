import subprocess

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def img_preprocess_sario(frame):
    img = frame.copy()

    # Resize the image
    scale_percent = 300  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    # Normalize channels to (0,255)
    norm = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)

    # Denoise the image with fastNL
    img = cv2.fastNlMeansDenoisingColored(img, None, 12, 10, 7, 15)

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform adaptive histogram equalization to enchance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)
    # Convert to binary using adaptive threshold
    # to keep as much of the number as possible
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )

    # Invert the image to have black foreground and white background
    img = cv2.bitwise_not(img)

    return img


def check_tesseract():
    try:
        # Try to run Tesseract with the --version command
        subprocess.run(
            ["tesseract", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If Tesseract couldn't be run, raise an exception
        raise RuntimeError("Tesseract is not installed or not accessible")


def extract_timestamp(frame):
    # Check that tesseract is installed and can be used
    check_tesseract()

    # Define the region of interest (ROI) coordinates
    start_x, start_y, end_x, end_y = (
        0,
        0,
        int(0.32 * frame.shape[1]),
        int(0.05 * frame.shape[0]),
    )

    # Crop the image to the ROI
    roi = frame[start_y:end_y, start_x:end_x]

    # Create a mask for pixels with green value
    # above 200 and red/blue below 100
    mask = (roi[:, :, 1] > 200) & (roi[:, :, 0] < 100) & (roi[:, :, 2] < 100)

    # Apply the mask to the ROI
    roi = cv2.bitwise_and(roi, roi, mask=mask.astype(np.uint8) * 255)

    # Convert the ROI to grayscale inverted
    inverted_roi = cv2.bitwise_not(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    # Convert the image to binary using Otsu's binarization
    _, binary_roi = cv2.threshold(inverted_roi, 0, 255, cv2.THRESH_OTSU)

    # Apply Gaussian & Median blur to get rid of pixeling
    blurred_roi = cv2.GaussianBlur(binary_roi, (5, 3), 0)
    blurred_roi = cv2.medianBlur(blurred_roi, 1)

    # Perform OCR using Tesseract psm 7 for single line text
    data = pytesseract.image_to_string(
        blurred_roi, lang="eng", config="--psm 7", output_type=Output.DICT
    )

    return data["text"]


def extract_sarionum(frame):
    # Check that tesseract is installed and can be used
    check_tesseract()

    # Run preprocessing pipeline on the image
    img = img_preprocess_sario(frame)

    # We use psm 10 (single character of text)
    # oem 1 (LSTM Algo)
    # and a whitelist of numbers
    data = pytesseract.image_to_string(
        img,
        lang="eng",
        config="--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789",
        output_type=Output.DICT,
    )

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    return data["text"]
