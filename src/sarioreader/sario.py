from ultralytics import YOLO
import numpy as np


class SarioDetector:
    def __init__(self, model_path):
        # Load the model
        self.model = YOLO(model_path)

    def detect_sario(self, frame):
        # Detect objects
        results = self.model(frame, device="cpu", conf=0.65, verbose=False)

        return results

    def extract_roi(self, frame, results):
        # Extract XY coords for each detected sario in the results
        roi = np.array([])
        for result in results:
            for box in result.boxes.xyxy:
                # Convert tensor to numpy
                box = box.numpy()

                # Add an extra ndim to make array 2D in case only one sario is found
                if box.ndim == 1:
                    box = box[np.newaxis, :]

                for bbox in box:
                    start_x, start_y, end_x, end_y = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )
                    # Make the ROI 2x bigger
                    height, width, _ = frame.shape
                    start_x = max(0, start_x - (end_x - start_x) // 2)
                    start_y = max(0, start_y - (end_y - start_y) // 2)
                    end_x = min(width, end_x + (end_x - start_x) // 2)
                    end_y = min(height, end_y + (end_y - start_y) // 2)

                    roi = frame[start_y:end_y, start_x:end_x]
        return roi
