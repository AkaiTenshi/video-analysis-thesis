import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import subprocess
from clearvision.ocr import OCR
from clearvision.imageproc.toolkit import adjust_contrast_brightness


class srOCR:
    def __init__(self):
        self.ocr_agent = OCR(scale_factor=2)

    def img_preprocess_sario(self, frame):
        return adjust_contrast_brightness(frame, "clahe")

    def check_tesseract(self):
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

    def extract_timestamp(self, frame):
        # Check that tesseract is installed and can be used
        self.check_tesseract()

        # Define the region of interest (ROI) coordinates
        start_x, start_y, end_x, end_y = (
            0,
            0,
            int(0.32 * frame.shape[1]),
            int(0.05 * frame.shape[0]),
        )

        # Crop the image to the ROI
        roi = frame[start_y:end_y, start_x:end_x]

        # Create a mask for pixels with green value above 200 and red/blue below 100
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

    def _sanitize_sarionum(self, text_list):
        sanitized_list = []
        substitution_map = {
            "O": "0",
            "I": "1",
            "S": "5",
            "Z": "2",
            "G": "6",
            "o": "0",
            "i": "1",
            "s": "5",
            "z": "2",
            "g": "6",
        }

        for text in text_list:
            sanitized_text = "".join(
                [substitution_map.get(char, char) for char in text]
            )
            if sanitized_text.isdigit():
                sanitized_list.append(sanitized_text)

        return sanitized_list

    def extract_sarionum(self, frame):
        # Check that tesseract is installed and can be used
        data = self.ocr_agent.perform_ocr(frame)

        if data is not None:
            raw_sarionum = [result["text"] for result in data.get("ocr_result", [])]
            return self._sanitize_sarionum(raw_sarionum)
        else:
            print("Could not find OCR data")
            return []
