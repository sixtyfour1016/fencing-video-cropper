import cv2
import random
import easyocr
import scoreboard
from scoreboard import Scoreboard

VIDEO_FILE = "input_videos/fencing_livestream_1.mp4"

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
FRAME_SKIP = 1500

scoreboard = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue

    scoreboard = Scoreboard(frame)

    GREEN_ROI_TOP_LEFT = scoreboard.GREEN_ROI_TOP_LEFT
    GREEN_ROI_BOTTOM_RIGHT = scoreboard.GREEN_ROI_BOTTOM_RIGHT
    RED_ROI_TOP_LEFT = scoreboard.RED_ROI_TOP_LEFT
    RED_ROI_BOTTOM_RIGHT = scoreboard.RED_ROI_BOTTOM_RIGHT

    print(f"Frame {frame_count}:")

    scoreboard.updateGreenFencerInfo()
    scoreboard.updateRedFencerInfo()

    green_fencer = scoreboard.green_fencer
    red_fencer = scoreboard.red_fencer

    cv2.rectangle(frame, GREEN_ROI_TOP_LEFT, GREEN_ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
    cv2.rectangle(frame, RED_ROI_TOP_LEFT, RED_ROI_BOTTOM_RIGHT, (0, 0, 255), 2)
    cv2.imshow("Scoreboard ROI", frame)

    scoreboard.printScoreboard()

    if cv2.waitKey(5) & 0xFF == ord(''):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
