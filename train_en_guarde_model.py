import cv2
import mediapipe as mp
import os

# Set paths
video_path = 'fencing_clip_102.mp4'
output_folder = 'output_data'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Frame counter and label data
frame_count = 0
pose_labels = []

def save_label(frame_number, label):
    """Save the label to a file."""
    with open(os.path.join(output_folder, 'pose_labels.txt'), 'a') as file:
        file.write(f"{frame_number},{label}\n")

# Loop through the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Pose Detection - Frame', frame)

    # Wait for user input (0 means wait indefinitely for a keypress)
    key = cv2.waitKey(0)

    if key == 38:  # Up arrow key for "en guarde"
        print(f"Frame {frame_count}: en guarde")
        save_label(frame_count, 'en_guarde')
    elif key == 39:  # Right arrow key for skipping (not en guarde)
        print(f"Frame {frame_count}: not en guarde")
        save_label(frame_count, 'not_en_guarde')

    # Exit the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
pose.close()
