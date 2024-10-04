import cv2
import numpy as np
import mediapipe as mp

class PoseDetector:
    def __init__(self, video_path, yolo_weights, yolo_cfg, coco_names):
        self.video_path = video_path
        self.pose_estimators = []
        self.pose_estimators_dim = []

        # Initialize YOLO
        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load class names (COCO dataset)
        with open(coco_names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.drawing_utils = mp.solutions.drawing_utils

        # Open video capture
        self.cap = cv2.VideoCapture(self.video_path)

    def compare_distance(self, dim1, dim2):
        """Calculate a distance metric to compare bounding boxes."""
        x1, y1, w1, h1 = dim1
        x2, y2, w2, h2 = dim2
        return np.linalg.norm(np.array([x1, y1, w1, h1]) - np.array([x2, y2, w2, h2]))

    def detect_and_track_pose(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            height, width, _ = frame.shape

            # Prepare the frame for YOLO
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            boxes, confidences, class_ids = [], [], []

            # Process YOLO detections
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and self.classes[class_id] == "person":
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

                    # Select pose estimator
                    selected_pose_idx = self.assign_pose_estimator([x, y, w, h])

                    # Get pose from selected pose estimator and process it
                    pose = self.pose_estimators[selected_pose_idx]
                    cropped_frame = frame[y:y + h, x:x + w]

                    if cropped_frame.size != 0:
                        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(rgb_frame)

                        if results.pose_landmarks:
                            self.drawing_utils.draw_landmarks(
                                cropped_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                            )

                    # Draw the bounding box on the original frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame with pose detection
            cv2.imshow('Pose Detection with YOLO', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def assign_pose_estimator(self, detected_dim):
        """Assign a pose estimator to the detected person."""
        if len(self.pose_estimators) == 0:
            # Create the first pose estimator
            pose = self.mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.7)
            self.pose_estimators.append(pose)
            self.pose_estimators_dim.append(detected_dim)
            return len(self.pose_estimators) - 1
        else:
            # Compare with existing estimators
            threshold_for_new = 100
            prev_high_score = 0
            selected_pose_idx_high = 0
            prev_low_score = 1000000000
            selected_pose_idx_low = 0

            for pose_idx, dim in enumerate(self.pose_estimators_dim):
                score = self.compare_distance(dim, detected_dim)
                if score > prev_high_score:
                    selected_pose_idx_high = pose_idx
                    prev_high_score = score
                if score < prev_low_score:
                    selected_pose_idx_low = pose_idx
                    prev_low_score = score
            
            if prev_high_score > threshold_for_new:
                # Create a new pose estimator for a new person
                pose = self.mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.7)
                self.pose_estimators.append(pose)
                self.pose_estimators_dim.append(detected_dim)
                return len(self.pose_estimators) - 1
            else:
                self.pose_estimators_dim[selected_pose_idx_low] = detected_dim
                return selected_pose_idx_low

# Example usage
if __name__ == "__main__":
    video_path = 'C:/Users/STT015/PycharmProjects/fencing-video-cropper/input_videos/fencing_clip_47.mp4'
    yolo_weights = "yolov4.weights"
    yolo_cfg = "yolov4.cfg"
    coco_names = "coco.names"

    pose_detector = PoseDetector(video_path, yolo_weights, yolo_cfg, coco_names)
    pose_detector.detect_and_track_pose()
