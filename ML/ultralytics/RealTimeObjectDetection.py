# Colab-compatible YOLOv8 simplified real-time annotation

import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from shapely.geometry import Point, Polygon
from IPython.display import Video

class YoloRealtimeUsecaseRunner:
    def __init__(self, model_path, use_case_conf):
        # Load the YOLOv8 model and store configuration dictionaries
        self.model = YOLO(model_path)
        self.annotation_config = use_case_conf 
        self.job_config = {}
        self.video_paths = {}
        self.person_tracking = {}

    def start_job(self, job_id, video_path, job_config):
        # Save job configuration for future reference
        self.job_config[job_id] = job_config
        use_cases = job_config.get("use_cases", [])

        # Set up video capture and output writer
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.video_paths[job_id] = output_path

        # Process each frame from the input video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            
            # Run each selected use case on the frame
            if "occupancy_violation" in use_cases:
                config = self.annotation_config.get("occupancy_violation", {})
                frame = self.occupancy_violation(frame, results, config)
            if "safety_check" in use_cases:
                config = self.annotation_config.get("safety_check", {})
                frame = self.safety_check(frame, results, config)
            if "roi_monitoring" in use_cases:
                config = self.annotation_config.get("roi_monitoring", {})
                frame = self.roi_monitoring(job_id, frame, results, config)
      
            writer.write(frame)

        # Release resources
        cap.release()
        writer.release()
        return output_path

    def _draw_box_with_label(self, frame, bbox, label, color):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def _draw_text_with_background(self, frame, text, position, text_color=(0, 0, 255), bg_color=(255, 255, 255),
                                font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, thickness=2, margin=5):
        """
        Draw text with a filled background rectangle on the frame.
        """
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - margin, y - text_height - margin),
            (x + text_width + margin, y + baseline + margin),
            bg_color,
            thickness=-1
        )
        # Draw the text
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA
        )
        return frame

    def occupancy_violation(self, frame, results, config):
        persons = []
        min_conf = config.get("minConf", 0.6)
        draw_labels = config.get("draw_labels", ["person"])
        class_names = results[0].names if hasattr(results[0], 'names') else results[0].model.names

        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6].cpu().numpy()
            class_name = class_names[int(cls)]
            
            if class_name in draw_labels and conf >= min_conf:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                persons.append({
                    "center": (cx, cy),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf
                })

        violation_indices = set()
        distance_threshold = config.get("distance_threshold", 50)
        overlap_threshold = config.get("overlap_threshold", 0.2)

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                pi, pj = persons[i], persons[j]
                dist = np.linalg.norm(np.array(pi["center"]) - np.array(pj["center"]))
                iou = self._iou(pi["bbox"], pj["bbox"])

                if dist <= distance_threshold and iou >= overlap_threshold:
                    violation_indices.update([i, j])

        # Draw annotations
        for idx, person in enumerate(persons):
            label = "occupancy_violation" if idx in violation_indices else "person"
            color = config.get("invalid_color" if idx in violation_indices else "valid_color", (0, 255, 0))
            frame = self._draw_box_with_label(frame, person["bbox"], "person", color)

        return frame

    def safety_check(self, frame, results, config):
        draw_labels = config.get("draw_labels", ["gloves", "no-gloves", "vest", "no-vest"])
        min_conf = config.get("minConf", 0.6)
        class_names = results[0].names if hasattr(results[0], 'names') else results[0].model.names

        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6].cpu().numpy()
            cls = int(cls)
            label = class_names[cls]

            if label not in draw_labels or conf < min_conf:
                continue

            color = config.get("invalid_color", (0, 0, 255)) if label.startswith("no-") else config.get("valid_color", (0, 255, 0))
            text = f"{label} ({conf:.2f})"
            frame = self._draw_box_with_label(frame, (int(x1), int(y1), int(x2), int(y2)), text, color)

        return frame

    def roi_monitoring(self, job_id, frame, results, config):
        roi_points = config.get("roi_polygon", [])
        if not roi_points:
            return frame

        roi_poly = Polygon(roi_points)
        min_conf = config.get("minConf", 0.6)
        draw_labels = config.get("draw_labels", ["person"])
        class_names = results[0].names if hasattr(results[0], 'names') else results[0].model.names

        present = False
        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box[:6].cpu().numpy()
            label = class_names[int(cls)]
            if label in draw_labels and conf >= min_conf:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if roi_poly.contains(Point(cx, cy)):
                    present = True
                    break

        # FPS from stored config (assumes you set it earlier per job)
        fps = self.job_config[job_id].get("fps", 1)

        # Track away time using frame count
        if job_id not in self.person_tracking:
            self.person_tracking[job_id] = {"away_frames": 0}

        if present:
            self.person_tracking[job_id]["away_frames"] = 0
            shade_color = (*config.get("valid_color", (0, 255, 0)), 25)
        else:
            self.person_tracking[job_id]["away_frames"] += 1
            away_seconds = int(self.person_tracking[job_id]["away_frames"] / fps)
            shade_color = (*config.get("invalid_color", (0, 0, 255)), 50)
            frame = self._draw_text_with_background(
                frame,
                text=f"Away: {away_seconds}s",
                position=(10, 30),
                text_color=config.get("invalid_color", (0, 0, 255)),
                bg_color=(255, 255, 255)
            )
        # Draw shaded ROI polygon
        overlay = frame.copy()
        roi_np = np.array(roi_points, dtype=np.int32)
        cv2.fillPoly(overlay, [roi_np], shade_color[:3])
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.polylines(frame, [roi_np], isClosed=True, color=shade_color[:3], thickness=2)

        return frame

    def _iou(self, boxA, boxB):
        # Calculate Intersection over Union (IoU) between two boxes
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def show_video(self, job_id):
        # Display annotated video in Colab
        return Video(self.video_paths[job_id])

if __name__ == "__main__":
    config = {
        "occupancy_violation": {
            "valid_color": (0, 255, 0),
            "invalid_color": (0, 0, 255),
            "distance_threshold": 50,
            "overlap_threshold": 0.2,
            "draw_labels": ["person"],
            "min_conf": 0.6
        },
        "safety_check": {
            "valid_color": (0, 165, 255),
            "invalid_color": (0, 0, 255),
            "overlap_threshold": 0.1,
            "draw_labels": ["gloves", "no-gloves", "vest", "no-vest"],
            "min_conf": 0.2
        },
        "roi_monitoring": {
            "valid_color": (0, 255, 0),
            "invalid_color": (0, 0, 255),
            "roi_polygon": [(1017, 124), (1000, 1060), (1886, 1054), (1907, 101), (1017, 124)]
        }
    } 

    job_config = {
        "use_cases": ["occupancy_violation", "safety_check", "roi_monitoring"]
        }
    detector = YoloRealtimeUsecaseRunner("best.pt", config)
    detector.start_job("PPE4", "PPE4.mp4", job_config)
    job_config = {
        "use_cases": ["occupancy_violation", "safety_check"]
        }
    detector.start_job("PPE1", "PPE1.mp4", job_config)
    detector.start_job("PPE2", "PPE2.mp4", job_config)
    detector.start_job("PPE3", "PPE3.mp4", job_config)
    detector.start_job("PPE5", "PPE5.mp4", job_config)
    

