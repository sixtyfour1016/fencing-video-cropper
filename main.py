import cv2
import numpy as np
import mediapipe as mp

# Load YOLO (using YOLOv4 for this example)
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names (from COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load the video
input_video_path = 'C:/Users/STT015/PycharmProjects/fencing-video-cropper/input_videos/fencing_clip_170.mp4'
cap = cv2.VideoCapture(input_video_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Global variables for pose tracking
pose_estimator = []
pose_estimator_dim = []

def compareDist(dim1, dim2):
    """Calculate a distance metric to compare bounding boxes."""
    x1, y1, w1, h1 = dim1
    x2, y2, w2, h2 = dim2
    return np.linalg.norm(np.array([x1, y1, w1, h1]) - np.array([x2, y2, w2, h2]))

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

    # Apply non-maxima suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Assign pose estimators to each detected person
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            # Determine which pose estimator to use
            selected_pose_idx = 0
            if len(pose_estimator) == 0:
                # Create a new pose estimator
                pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.6)
                pose_estimator.append(pose)
                pose_estimator_dim.append([x, y, w, h])
                selected_pose_idx = len(pose_estimator) - 1
            else:
                threshold_for_new = 100
                prev_high_score = 0
                selected_pose_idx_high = 0
                prev_low_score = 1000000000
                selected_pose_idx_low = 0
                
                for pose_idx, dim in enumerate(pose_estimator_dim):
                    score = compareDist(dim, [x, y, w, h])
                    if score > prev_high_score:
                        selected_pose_idx_high = pose_idx
                        prev_high_score = score
                    if score < prev_low_score:
                        selected_pose_idx_low = pose_idx
                        prev_low_score = score

                if prev_high_score > threshold_for_new:
                    # Create a new pose estimator for a new person
                    pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.6)
                    pose_estimator.append(pose)
                    pose_estimator_dim.append([x, y, w, h])
                    selected_pose_idx = len(pose_estimator) - 1
                else:
                    selected_pose_idx = selected_pose_idx_low
                    pose_estimator_dim[selected_pose_idx] = [x, y, w, h]

            # Get the selected pose estimator and apply it to the cropped person frame
            pose = pose_estimator[selected_pose_idx]
            cropped_frame = frame[y:y + h, x:x + w]

            if cropped_frame.size != 0:
                # Convert the frame to RGB (for MediaPipe)
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                # Draw pose landmarks on the frame if detected
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(cropped_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw the bounding box on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with pose detection
    cv2.imshow('Pose Detection with YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
