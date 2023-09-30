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
    # Desired time in seconds
    desired_time_seconds = 40  # Adjust this to your desired time

    # Calculate the frame number corresponding to the desired time
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    desired_frame = int(desired_time_seconds * frame_rate)

    # Set the video's position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, desired_frame)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            results = detector.detect_sario(frame)
            roi = detector.extract_roi(frame, results)

            if roi.any():
                roi = ocr.img_preprocess_sario(roi)
                sarionum = ocr.extract_sarionum(roi)
                timestamp = ocr.extract_timestamp(frame)
                cv2.imshow("Debug", roi)
                cv2.waitKey(1)

                if sarionum:
                    logger.info(
                        f"Found Sario with numbers {' '.join(map(str, sarionum))} on {timestamp}"  # noqa: E501
                    )
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file")

    args = parser.parse_args()
    main(args.video_path)
