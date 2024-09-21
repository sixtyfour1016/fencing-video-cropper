import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.9
)

mp_drawing = mp.solutions.drawing_utils

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
input_video_path = 'C:/Users/STT015/PycharmProjects/fencing-video-cropper/input_videos/sample_video.mp4'  # Change this to your input video file path
cap = cv2.VideoCapture(input_video_path)

# Define the enlargement factor for bounding boxes
enlargement_factor = 1.2  # Increase this value to make boxes bigger

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

                # Increase the size of the bounding box
                # w = int(w * enlargement_factor)
                # h = int(h * enlargement_factor)
                # x = max(0, int(center_x - w / 2))  # Adjust x to stay within bounds
                # y = max(0, int(center_y - h / 2))  # Adjust y to stay within bounds

                # Ensure the bounding box is within the frame
                if x >= 0 and y >= 0 and (x + w) <= width and (y + h) <= height:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply non-maxima suppression (NMS) to avoid multiple boxes for the same object
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes for detected people and apply MediaPipe Pose
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green for people
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Crop the detected person's area from the frame
            cropped_frame = frame[y:y+h, x:x+w]

            # Ensure the cropped frame is not empty
            if cropped_frame.size == 0:
                continue

            # Normalize the cropped frame for better lighting contrast
            normalized_frame = cv2.normalize(cropped_frame, None, 0, 255, cv2.NORM_MINMAX)

            # Convert the normalized frame to RGB for MediaPipe
            image_rgb = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2RGB)

            # Run MediaPipe Pose on the original size cropped frame
            results = pose.process(image_rgb)

            # Draw pose landmarks if they exist
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(cropped_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Place the processed frame back into the original frame
            frame[y:y+h, x:x+w] = cropped_frame

    # Display the resulting frame with bounding boxes and pose landmarks
    cv2.imshow('Fencing Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
