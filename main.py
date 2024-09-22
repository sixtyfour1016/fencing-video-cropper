import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

if len(output_layers_indices) == 0:
    print("No output layers found.")
else:
    output_layers = [layer_names[i - 1] for i in output_layers_indices.flatten()]

# Load class names (from COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the video
input_video_path = 'C:/Users/STT015/PycharmProjects/fencing-video-cropper/input_videos/sample_video_3.mp4'  # Change this to your input video file path
cap = cv2.VideoCapture(input_video_path)

def create_gaussian_mask(shape):
    """
    Create a Gaussian mask that gives more weight to the center of the image.
    """
    center_x, center_y = shape[1] // 2, shape[0] // 2
    sigma = min(shape) // 6  # Standard deviation, controls the spread of the weighting
    mask_x, mask_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    gaussian_mask = np.exp(-((mask_x - center_x) ** 2 + (mask_y - center_y) ** 2) / (2 * sigma ** 2))
    return gaussian_mask / gaussian_mask.max()  # Normalize between 0 and 1


def analyze_color(frame):
    """
    Analyze the dominant color in the frame, giving more weight to the center.
    Returns True if it's more likely a fencer (white/metallic) and False if a referee (black).
    """
    # Convert frame to HSV (to better distinguish colors)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for detecting white (fencers)
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])

    # Define color range for detecting metallic (reflective clothing)
    lower_metallic = np.array([0, 0, 200])
    upper_metallic = np.array([180, 50, 255])

    # Define color range for detecting black (referees)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Create masks for each color
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_metallic = cv2.inRange(hsv, lower_metallic, upper_metallic)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Create a Gaussian weight mask
    gaussian_mask = create_gaussian_mask(frame.shape[:2])

    # Apply the weight to each mask
    white_weighted = np.sum(mask_white * gaussian_mask)
    metallic_weighted = np.sum(mask_metallic * gaussian_mask)
    black_weighted = np.sum(mask_black * gaussian_mask)

    # If there are more weighted black pixels, it's likely a referee
    if black_weighted > white_weighted and black_weighted > metallic_weighted:
        return False  # Referee
    else:
        return True  # Fencer


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Ensure the bounding box is within the frame
                if x >= 0 and y >= 0 and (x + w) <= width and (y + h) <= height:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply non-maxima suppression (NMS) to avoid multiple boxes for the same object
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes for detected people and apply color filtering
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            # Crop the detected person's area from the frame
            cropped_frame = frame[y:y + h, x:x + w]

            # Ensure the cropped frame is not empty
            if cropped_frame.size == 0:
                continue

            # Analyze the color in the bounding box with center-weighted analysis
            if analyze_color(cropped_frame):  # Likely a fencer
                color = (0, 255, 0)  # Green for fencers
            else:  # Likely a referee
                color = (0, 0, 255)  # Red for referees

            # Draw the bounding box with the detected color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the resulting frame with bounding boxes
    cv2.imshow('Fencing Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
