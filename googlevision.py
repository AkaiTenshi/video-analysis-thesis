import cv2
import numpy as np
from google.cloud import vision


def detect_text_from_image(image):
    img = image.copy()
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

    # Convert the image to bytes
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    byte_im = im_buf_arr.tobytes()

    # Create a new client for vision
    client = vision.ImageAnnotatorClient.from_service_account_json(
        "api-key.json"
        )

    # Create a new image object
    image = vision.Image(content=byte_im)

    # Use the client to detect text from the image
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Get the entire detected text
    if len(texts) > 0:
        detected_text = texts[0].description
    else:
        detected_text = ""

    return detected_text.strip()
