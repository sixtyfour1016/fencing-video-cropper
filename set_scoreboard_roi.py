import cv2

# Video file path
VIDEO_FILE = "input_videos/fencing_livestream_3.mp4"

# Define initial ROI coordinates (manually adjust these as needed)
center_x = 640
center_y = 600
x_offset = 500
y_offset = 40
ROI_TOP_LEFT = (center_x - x_offset, center_y - y_offset)   # (x, y) top-left corner
ROI_BOTTOM_RIGHT = (center_x + x_offset, center_y + y_offset)  # (x, y) bottom-right corner

# Open the video file
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f}s")

frame_count = 0
FRAME_SKIP = 500  # Process every 30th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for faster processing
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue

    # Draw the ROI box
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)  # Green box

    # Display the frame
    cv2.imshow("ROI Selection - Adjust the Coordinates", frame)

    # Break on 'q' key press
    key = cv2.waitKey(30)
    if key == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
