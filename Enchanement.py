import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time


def mediapipe_detection(image, model):
    """
    Process an image through MediaPipe Holistic model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """
    Draw landmarks for pose, face, and hands on the image.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )

    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    # Draw left hand landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # Draw right hand landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    """
    Extract keypoints from MediaPipe results.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


class SignLanguageProcessor:
    def __init__(self, sign_model_path='action.h5', confidence_threshold=0.5):
        # Load the sign detection model
        try:
            self.sign_model = load_model(sign_model_path)
            print("Successfully loaded sign language model")
        except Exception as e:
            print(f"Error loading sign model: {e}")
            raise

        self.confidence_threshold = confidence_threshold

        # Load T5 model for text enhancement
        print("Loading T5 model and tokenizer...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
            self.nlp_model = T5ForConditionalGeneration.from_pretrained('t5-small')
            print("Successfully loaded T5 model")
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            raise

        self.context_window = []
        self.max_context_size = 10
        self.last_process_time = time.time()
        self.min_process_interval = 1.0  # Minimum time between processing in seconds

        # Common sign combinations and their translations
        self.sign_phrases = {
            ('hello', 'hello'): 'Hello',
            ('thanks', 'thanks'): 'Thank you',
            ('iloveyou', 'iloveyou'): 'I love you'
        }

    def _should_process(self):
        """Rate limiting check"""
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return False
        self.last_process_time = current_time
        return True

    def _remove_duplicates(self, signs):
        """Remove consecutive duplicates while preserving order"""
        if not signs:
            return []
        result = [signs[0]]
        for s in signs[1:]:
            if s != result[-1]:
                result.append(s)
        return result

    def process_signs(self, detected_signs):
        """Process detected signs into meaningful sentences"""
        if not detected_signs or not self._should_process():
            return ""

        # Extract signs and confidences
        signs = [sign for sign, conf in detected_signs if conf > self.confidence_threshold]

        # Remove consecutive duplicates
        unique_signs = self._remove_duplicates(signs)

        # Check for known sign combinations
        sign_tuple = tuple(unique_signs)
        if sign_tuple in self.sign_phrases:
            return self.sign_phrases[sign_tuple]

        # Convert to sentence
        raw_sentence = " ".join(unique_signs)

        # Enhance with T5
        input_text = f"convert to natural sentence: {raw_sentence}"
        try:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            outputs = self.nlp_model.generate(
                input_ids,
                max_length=50,
                num_beams=4,
                temperature=0.7,
                do_sample=False  # Disable sampling for more consistent output
            )
            enhanced_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean up any remaining artifacts
            enhanced_sentence = enhanced_sentence.replace("convert to natural sentence:", "").strip()

        except Exception as e:
            print(f"Error in T5 processing: {e}")
            enhanced_sentence = raw_sentence

        return enhanced_sentence


def main():
    # Initialize processor
    try:
        processor = SignLanguageProcessor()
        print("SignLanguageProcessor initialized successfully")
    except Exception as e:
        print(f"Failed to initialize SignLanguageProcessor: {e}")
        return

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    actions = np.array(['thanks', 'hello', 'iloveyou'])

    # Initialize detection variables
    sequence = []
    sentence_buffer = []
    threshold = 0.5

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Starting real-time detection...")

    last_detection_time = time.time()
    min_detection_interval = 0.5  # Minimum time between detections

    # Set up MediaPipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            current_time = time.time()
            if current_time - last_detection_time >= min_detection_interval:
                # Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep only last 30 frames

                if len(sequence) == 30:
                    res = processor.sign_model.predict(np.expand_dims(sequence, axis=0))[0]

                    if res[np.argmax(res)] > threshold:
                        detected_sign = actions[np.argmax(res)]
                        confidence = float(res[np.argmax(res)])

                        # Add to buffer
                        sentence_buffer.append((detected_sign, confidence))

                        # Process when we have enough signs
                        if len(sentence_buffer) >= 2:  # Process after 2 signs
                            enhanced_sentence = processor.process_signs(sentence_buffer)
                            print(f"Raw signs: {sentence_buffer}")
                            print(f"Enhanced: {enhanced_sentence}")

                            # Display on frame
                            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                            cv2.putText(image, enhanced_sentence, (3, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            sentence_buffer = []  # Clear buffer

                    last_detection_time = current_time

            # Show frame
            cv2.imshow('OpenCV Feed', image)

            # Break loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Sign Language Enhancement System...")
    main()