from ocr import extract_sarionum, extract_timestamp
from sario import SarioDetector
import cv2

# Open the video file
cap = cv2.VideoCapture('number.mp4')
detector = SarioDetector('models/sario-best.pt')
fps = cap.get(cv2.CAP_PROP_FPS)
start_time_in_seconds = 30

start_frame_number = int(start_time_in_seconds * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

# Open a text file to write the timestamps
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Use the detector to find objects
        results = detector.detect_sario(frame)
        roi = detector.extract_roi(frame, results)
        
        if roi.any():
            sarionum = extract_sarionum(roi)
            timestamp = extract_timestamp(frame)
            
            print("Found Sario with number " + sarionum + " on " + timestamp)
                                          
    else:
        # If no frame could be read (e.g. end of video), break the loop
        break

cap.release()
cv2.destroyAllWindows()