import cv2
import easyocr

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])

# Define video path and ROI
VIDEO_FILE = "input_videos/fencing_livestream_3.mp4"

# Define initial ROI coordinates (manually adjust these as needed) 720p
center_x = 640
center_y = 600
x_offset = 500
y_offset = 40
ROI_TOP_LEFT = (center_x - x_offset, center_y - y_offset)   # (x, y) top-left corner
ROI_BOTTOM_RIGHT = (center_x + x_offset, center_y + y_offset)  # (x, y) bottom-right corner

# Open the video
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
FRAME_SKIP = 1000  # Process every 30th frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for faster processing
    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue

    # Crop the ROI (scoreboard region)
    scoreboard_roi = frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]

    # Preprocess the image for better OCR results:
    # Convert to grayscale
    gray_scoreboard = cv2.cvtColor(scoreboard_roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (binary image) to improve contrast
    _, thresh_scoreboard = cv2.threshold(gray_scoreboard, 150, 255, cv2.THRESH_BINARY_INV)

    # Optional: Resize the image to make text larger
    resized_scoreboard = cv2.resize(thresh_scoreboard, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Perform OCR with EasyOCR on the preprocessed image
    results = reader.readtext(resized_scoreboard)

    # Print detected text
    print(f"Frame {frame_count}:")
    for (bbox, text, prob) in results:
        print(f"Detected Text: {text} (Confidence: {prob:.2f})")

    # Display the video frame with the ROI box
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
    cv2.imshow("Scoreboard ROI", frame)
    cv2.imshow("Processed Scoreboard", resized_scoreboard)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
