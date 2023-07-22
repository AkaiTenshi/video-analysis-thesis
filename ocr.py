import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import subprocess

def check_tesseract():
    try:
        # Try to run Tesseract with the --version command
        subprocess.run(["tesseract", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If Tesseract couldn't be run, raise an exception
        raise RuntimeError("Tesseract is not installed or not accessible")

def extract_timestamp(frame):
    
    #Check that tesseract is installed and can be used
    check_tesseract()
    
    # Define the region of interest (ROI) coordinates
    start_x, start_y, end_x, end_y = 0, 0, int(0.32*frame.shape[1]), int(0.05*frame.shape[0])

    # Crop the image to the ROI
    roi = frame[start_y:end_y, start_x:end_x]
    
    # Create a mask for pixels with green value above 200 and red/blue below 100
    mask = (roi[:,:,1] > 200) & (roi[:,:,0] < 100) & (roi[:,:,2] < 100)

    # Apply the mask to the ROI
    roi = cv2.bitwise_and(roi, roi, mask=mask.astype(np.uint8)*255)
    
    # Convert the ROI to grayscale inverted
    inverted_roi = cv2.bitwise_not(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    # Convert the image to binary using Otsu's binarization
    _, binary_roi = cv2.threshold(inverted_roi, 0, 255, cv2.THRESH_OTSU)
    
    # Apply Gaussian & Median blur to get rid of pixeling
    blurred_roi = cv2.GaussianBlur(binary_roi, (5, 3), 0)
    blurred_roi = cv2.medianBlur(blurred_roi, 1)

    # Perform OCR using Tesseract psm 7 for single line text
    data = pytesseract.image_to_string(blurred_roi, lang='eng', config='--psm 7', output_type=Output.DICT)

    return data["text"]

def extract_sarionum(frame):
    #Check that tesseract is installed and can be used
    check_tesseract()
    
    #Convert to GRAYSCALE
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    scale_percent = 200 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_LINEAR)
    
    #Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    
    #Convert to binary using adaptive threshold to keep as much of the number as possible
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 4)
    
    #Dilate & Erode the image to fill gaps and reduce noise more
    kernel = np.ones((5,3),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations = 1)
    eroded = cv2.erode(dilated, kernel, iterations = 2)
    
    #Apply open and close morphology to remove "holes"    
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    #Apply sharpness
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(closed, -1, kernel)
    
    data = pytesseract.image_to_string(sharp, lang='eng', config='--psm 10', output_type=Output.DICT)

    
    return data['text']
       