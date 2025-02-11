import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


def initialize_mediapipe():
    """Initialize MediaPipe with basic settings"""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return mp_holistic, holistic


def process_frame(frame, holistic):
    """Process a single frame"""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame
    results = holistic.process(rgb_frame)
    return results


def draw_landmarks(frame, results, mp_holistic, mp_drawing):
    """Draw landmarks on the frame"""
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

    # Draw left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

    # Draw right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def extract_keypoints(results):
    """Extract keypoints from results"""
    # Initialize arrays with zeros
    pose = np.zeros(33 * 4)
    face = np.zeros(468 * 3)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    # Extract pose landmarks if available
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten()

    # Extract face landmarks if available
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten()

    # Extract hand landmarks if available
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, face, lh, rh])


def main():
    try:
        # Initialize MediaPipe
        mp_holistic, holistic = initialize_mediapipe()
        mp_drawing = mp.solutions.drawing_utils

        # Load the model
        try:
            model = load_model('action1.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video capture")
            return

        # Initialize variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        actions = np.array(['thanks', 'help', 'you-good', 'please'])

        print("Starting video capture...")
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            try:
                # Process frame
                results = process_frame(frame, holistic)

                # Draw landmarks
                draw_landmarks(frame, results, mp_holistic, mp_drawing)

                # Extract keypoints
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep only last 30 frames

                if len(sequence) == 30:
                    # Make prediction
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    # Process prediction
                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                # Display results
                cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Show frame
                cv2.imshow('OpenCV Feed', frame)

            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

            # Break loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

    except Exception as e:
        print(f"Critical error: {e}")


if __name__ == "__main__":
    main()