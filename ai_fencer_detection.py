import cv2
import numpy as np
import mediapipe as mp

class PoseDetector:
    def __init__(self, video_path, yolo_weights, yolo_cfg, coco_names):
        self.video_path = video_path
        self.pose_estimators = []
        self.pose_estimators_dim = []
        self.reference_pose_landmarks = None  # Reference pose for en garde position
        self.timestamps = []  # Store the timestamps of "en garde" frames

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

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)  # Get original FPS of the video
        self.target_fps = 8  # Target FPS
        self.skip_interval = int(self.original_fps / self.target_fps)  # Calculate the skip interval

    def set_reference_pose(self, pose_landmarks):
        """Sets the reference pose for en garde position."""
        self.reference_pose_landmarks = pose_landmarks

    def compare_distance(self, dim1, dim2):
        """Calculate a distance metric to compare bounding boxes."""
        x1, y1, w1, h1 = dim1
        x2, y2, w2, h2 = dim2
        return np.linalg.norm(np.array([x1, y1, w1, h1]) - np.array([x2, y2, w2, h2]))

    def compare_poses(self, current_landmarks):
        """Compare the current pose to the reference en garde pose."""
        if not self.reference_pose_landmarks:
            return False  # No reference pose set

        threshold = 0.1  # Set a threshold for pose similarity (tune this)
        total_diff = 0

        # Compare each corresponding landmark
        for i in range(len(self.reference_pose_landmarks)):
            ref_landmark = self.reference_pose_landmarks[i]
            curr_landmark = current_landmarks[i]
            diff = np.linalg.norm([ref_landmark.x - curr_landmark.x, ref_landmark.y - curr_landmark.y])
            total_diff += diff

        avg_diff = total_diff / len(self.reference_pose_landmarks)

        # Check if the difference is within the threshold
        return avg_diff < threshold

    def detect_fencers(self, frame):
        """Detect fencers using YOLO and return bounding boxes."""
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

        # Return filtered boxes (top 2 largest by area, representing 2 fencers)
        if len(indices) > 0:
            filtered_boxes = [boxes[i] for i in indices.flatten()]
            filtered_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            return filtered_boxes[:2]  # Select top 2
        return []

    def track_pose(self, frame, boxes, frame_count):
        """Track the pose of the fencers using MediaPipe Pose."""
        for box in boxes:
            x, y, w, h = box

            # Select pose estimator
            selected_pose_idx = self.assign_pose_estimator([x, y, w, h])

            # Get pose from selected pose estimator and process it
            pose = self.pose_estimators[selected_pose_idx]
            cropped_frame = frame[y:y + h, x:x + w]

            if cropped_frame.size != 0:
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    # Draw the landmarks
                    self.drawing_utils.draw_landmarks(
                        cropped_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                    )

                    # Set reference pose if 'r' key is pressed
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('r'):
                        self.set_reference_pose(results.pose_landmarks.landmark)
                        print("Reference pose set for en garde position.")

                    # If reference pose is set, compare it with the current frame
                    current_landmarks = results.pose_landmarks.landmark
                    if self.reference_pose_landmarks:
                        if self.compare_poses(current_landmarks):
                            timestamp = frame_count
                            self.timestamps.append(timestamp)
                            print(f"En Garde pose detected at timestamp: {timestamp:.2f} ")

    def detect_and_track_pose(self):
        frame_count = 0  # Track the frame count for timestamp calculation

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1  # Increment frame count

            # Skip frames based on the calculated interval to achieve 8 FPS
            if frame_count % self.skip_interval != 0:
                continue

            # Step 1: Detect fencers
            boxes = self.detect_fencers(frame)

            # Step 2: Track pose of detected fencers
            if boxes:
                self.track_pose(frame, boxes, frame_count)

            # Display the frame with pose detection
            cv2.imshow('Pose Detection with YOLO', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def assign_pose_estimator(self, detected_dim):
        detection_confidence = 0.6
        tracking_confidence = 0.6
        complexity = 0

        """Assign a pose estimator to the detected person."""
        if len(self.pose_estimators) == 0:
            # Create the first pose estimator
            pose = self.mp_pose.Pose(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence, model_complexity=complexity)
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
                pose = self.mp_pose.Pose(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence, model_complexity=complexity)
                self.pose_estimators.append(pose)
                self.pose_estimators_dim.append(detected_dim)
                return len(self.pose_estimators) - 1
            else:
                self.pose_estimators_dim[selected_pose_idx_low]

# Example usage
if __name__ == "__main__":
    video_path = 'C:/Users/STT015/PycharmProjects/fencing-video-cropper/input_videos/fencing_clip_102.mp4'
    yolo_weights = "yolov4.weights"
    yolo_cfg = "yolov4.cfg"
    coco_names = "coco.names"

    pose_detector = PoseDetector(video_path, yolo_weights, yolo_cfg, coco_names)
    pose_detector.detect_and_track_pose()

    # Print the timestamps of all frames where "en garde" position was detected
    print("En Garde frames detected at:", pose_detector.timestamps)
