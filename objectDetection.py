import cv2
import torch
from picamera2 import Picamera2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov5n.pt"):
        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = 0.3  # Lower confidence for speed
        self.model.iou = 0.45  # Adjust IoU threshold for efficiency

        # Initialize PiCamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (320, 240), "format": "RGB888"})  # Lower resolution
        self.picam2.configure(config)
        self.picam2.start()
        print("Model and Camera initialized successfully!")

    def detect_and_draw(self, frame):
        # Convert frame to correct format
        results = self.model(frame, size=320)  # Reduce input size for faster inference

        detected_objects = []  # Store detected object names
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            label = f"{self.model.names[cls]} {conf:.2f}"
            detected_objects.append(self.model.names[cls])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detected_objects, frame  # Return detected objects & modified frame

    def run(self):
        print("Starting YOLOv5 detection on Raspberry Pi... Press 'q' to quit.")

        while True:
            frame = self.picam2.capture_array()
            detected_objects, frame = self.detect_and_draw(frame)
            print("Detected Objects:", detected_objects)

            cv2.imshow('YOLOv5 Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run()

