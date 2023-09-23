import argparse
import logging
import os

import cv2

from sarioreader import detector, logger, ocr


def main(video_path):
    logger.setLevel(logging.INFO)

    if not os.path.exists(video_path):
        logger.error(f"Video file {video_path} does not exist.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time_in_seconds = 220

    start_frame_number = int(start_time_in_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
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

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file")

    args = parser.parse_args()
    main(args.video_path)
