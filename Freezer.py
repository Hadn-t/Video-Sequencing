import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class SignLanguageSystem:
    def __init__(self, existing_actions, new_actions):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.actions = np.concatenate([existing_actions, new_actions])
        self.new_actions = new_actions
        self.DATA_PATH = os.path.join('MP_Data')
        self.no_sequences = 25
        self.sequence_length = 30

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION)
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS)

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def setup_folders(self):
        for action in self.actions:
            for sequence in range(self.no_sequences):
                os.makedirs(os.path.join(self.DATA_PATH, action, str(sequence)), exist_ok=True)

    def collect_data(self):
        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for action in self.new_actions:  # Only collect new actions
                for sequence in range(self.no_sequences):
                    for frame_num in range(self.sequence_length):
                        ret, frame = cap.read()
                        image, results = self.mediapipe_detection(frame, holistic)
                        self.draw_styled_landmarks(image, results)

                        if frame_num == 0:
                            cv2.putText(image, f'STARTING COLLECTION: {action}', (50, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image, f'Collecting {action} - Sequence {sequence}', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)

                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(self.DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

        cap.release()
        cv2.destroyAllWindows()

    def preprocess_data(self):
        sequences, labels = [], []
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                sequences.append(window)
                labels.append(self.actions.tolist().index(action))

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        return train_test_split(X, y, test_size=0.05)

    def load_and_update_model(self):
        try:
            base_model = load_model('action.h5')
            print("Loaded existing model.")

            # Create new model with same architecture but different output size
            new_model = Sequential([
                LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
                LSTM(128, return_sequences=True, activation='relu'),
                LSTM(64, return_sequences=False, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(len(self.actions), activation='softmax')
            ])

            # Copy weights for all layers except the last dense layer
            for new_layer, old_layer in zip(new_model.layers[:-1], base_model.layers[:-1]):
                new_layer.set_weights(old_layer.get_weights())

            new_model.compile(optimizer=Adam(learning_rate=0.001),
                              loss='categorical_crossentropy',
                              metrics=['categorical_accuracy'])

            return new_model

        except:
            print("No existing model found. Training from scratch.")
            return None

    def train_model(self, X_train, y_train):
        model = self.load_and_update_model()

        if model is None:
            model = Sequential([
                LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
                LSTM(128, return_sequences=True, activation='relu'),
                LSTM(64, return_sequences=False, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(len(self.actions), activation='softmax')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])

        model.fit(X_train, y_train,
                  epochs=200,
                  batch_size=32,
                  validation_split=0.2)

        return model

    def real_time_detection(self, model):
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        cap = cv2.VideoCapture(0)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_styled_landmarks(image, results)

                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                        if len(sentence) == 0 or self.actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(self.actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():
    # Define your actions
    existing_actions = np.array(['thanks', 'help', 'you-good'])
    new_actions = np.array(['please'])

    # Initialize the system
    sls = SignLanguageSystem(existing_actions, new_actions)

    # Setup folders and collect data
    sls.setup_folders()
    sls.collect_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = sls.preprocess_data()

    # Train model
    model = sls.train_model(X_train, y_train)

    # Save model
    model.save('action1.h5')

    # Start real-time detection
    sls.real_time_detection(model)


if __name__ == "__main__":
    main()