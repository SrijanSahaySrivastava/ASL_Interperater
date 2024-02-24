import cv2
import os
import numpy as np

import mediapipe as mp

from Mediapipe_detections import mediapipe_detection, draw_landmarks, mp_holistic, extract_keypoints


# Create a VideoCapture object
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        # draw landmarks
        draw_landmarks(image, results)

        print(extract_keypoints(results))
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow("frame", image)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    # cap.destroyAllWindows()
