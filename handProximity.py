import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2
from collections import deque


class HandDistanceDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,  # Lowered for speed
            min_tracking_confidence=0.5
        )

        # Distance smoothing buffer
        self.distance_buffer = deque(maxlen=5)

        # Constants for distance estimation
        self.PALM_WIDTH_METERS = 0.085  # Approx. width of a human palm in meters
        self.FOCAL_LENGTH_PIXELS = 500  # Reduced for better performance on Pi

        # Initialize PiCamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (320, 240), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        print("Camera Initialized!")

    def calculate_distance(self, hand_landmarks, frame_width, frame_height):
        """Estimate hand-to-camera distance using palm width."""
        index_mcp = hand_landmarks.landmark[5]  # Index finger base
        pinky_mcp = hand_landmarks.landmark[17]  # Pinky base

        # Convert landmarks to pixel coordinates
        index_x, index_y = int(index_mcp.x * frame_width), int(index_mcp.y * frame_height)
        pinky_x, pinky_y = int(pinky_mcp.x * frame_width), int(pinky_mcp.y * frame_height)

        # Calculate palm width in pixels
        palm_width_pixels = np.sqrt((index_x - pinky_x) ** 2 + (index_y - pinky_y) ** 2)
        if palm_width_pixels == 0:
            return None  # Avoid division by zero

        # Estimate distance
        distance = (self.PALM_WIDTH_METERS * self.FOCAL_LENGTH_PIXELS) / palm_width_pixels
        return distance

    def run(self):
        """Runs the hand tracking system."""
        print("Starting Hand Distance Detection... Press 'q' to quit.")

        while True:
            frame = self.picam2.capture_array()  # Capture frame
            frame = cv2.flip(frame, 1)  # Mirror effect for natural interaction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            frame_height, frame_width = frame.shape[:2]

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                distance = self.calculate_distance(hand_landmarks, frame_width, frame_height)

                if distance:
                    self.distance_buffer.append(distance)
                    avg_distance = sum(self.distance_buffer) / len(self.distance_buffer)  # Smoothing

                    # Determine color based on distance
                    if avg_distance < 0.3:
                        color = (0, 255, 0)  # Green (close)
                    elif avg_distance < 0.6:
                        color = (0, 255, 255)  # Yellow (medium)
                    else:
                        color = (0, 0, 255)  # Red (far)

                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                                   self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                                                   self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

                    # Display distance on the screen
                    cv2.putText(frame, f'Distance: {avg_distance:.2f}m', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Show the frame
            cv2.imshow('Hand Distance Detector', frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = HandDistanceDetector()
    detector.run()

