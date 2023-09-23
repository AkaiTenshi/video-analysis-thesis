import logging

import cv2

from sarioreader import detector, logger, ocr


def main():
    logger.setLevel(logging.INFO)

    cap = cv2.VideoCapture(
        "/home/chmaikos/HUA/video-analysis-thesis/number.mp4"
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time_in_seconds = 220

    start_frame_number = int(start_time_in_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Use the detector to find objects
            results = detector.detect_sario(frame)
            roi = detector.extract_roi(frame, results)

            if roi.any():
                roi = ocr.img_preprocess_sario(roi)
                sarionum = ocr.extract_sarionum(roi)
                timestamp = ocr.extract_timestamp(frame)

                logger.info(
                    f"Found Sario with numbers {' '.join(map(str, sarionum))} on {timestamp}"  # noqa: E501
                )
                cv2.imshow("View", roi)

                # Wait for 1 ms, break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # If no frame could be read (e.g. end of video), break the loop
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
