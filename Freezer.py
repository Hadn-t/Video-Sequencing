# # 1. Import dependencies
# import cv2
# import numpy as np
# import os
# from matplotlib import pyplot as plt
# import time
# import mediapipe as mp
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # 2. Keypoints using MP Holistic
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
#
#
# def mediapipe_detection(image, model):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     return image, results
#
#
# def draw_styled_landmarks(image, results):
#     mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#
# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
#                     ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#     face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
#                     ).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
#                   ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
#                   ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
#     return np.concatenate([pose, face, lh, rh])
#
#
# # Setup Folders for Collection
# DATA_PATH = os.path.join('MP_Data')
# existing_actions = np.array(['thanks'])
# new_actions = np.array(['help'])  # Add new words here
# actions = np.concatenate([existing_actions, new_actions])
#
# no_sequences = 25
# sequence_length = 30
#
# # Create folders for data collection
# for action in actions:
#     for sequence in range(no_sequences):
#         os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
#
#
# # Data Collection Function
# def collect_data():
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         for action in new_actions:  # Only collect new words
#             for sequence in range(no_sequences):
#                 for frame_num in range(sequence_length):
#                     ret, frame = cap.read()
#                     image, results = mediapipe_detection(frame, holistic)
#                     draw_styled_landmarks(image, results)
#
#                     # Display collection status
#                     if frame_num == 0:
#                         cv2.putText(image, f'STARTING COLLECTION: {action}', (50, 200),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
#                         cv2.imshow('OpenCV Feed', image)
#                         cv2.waitKey(2000)
#                     else:
#                         cv2.putText(image, f'Collecting {action}', (50, 50),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#                         cv2.imshow('OpenCV Feed', image)
#
#                     # Save keypoints
#                     keypoints = extract_keypoints(results)
#                     npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                     np.save(npy_path, keypoints)
#
#                     if cv2.waitKey(10) & 0xFF == ord('q'):
#                         break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # Preprocess Data
# def preprocess_data():
#     sequences, labels = [], []
#     for action in actions:
#         for sequence in range(no_sequences):
#             window = []
#             for frame_num in range(sequence_length):
#                 res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
#                 window.append(res)
#             sequences.append(window)
#             labels.append(actions.tolist().index(action))
#
#     X = np.array(sequences)
#     y = to_categorical(labels).astype(int)
#     return train_test_split(X, y, test_size=0.05)
#
#
# # Load existing model and update
# def load_and_update_model():
#     try:
#         model = load_model('action.h5')
#         print("Loaded existing model.")
#     except:
#         print("No existing model found. Training from scratch.")
#         return None
#
#     # Modify output layer with unique names
#     x = model.layers[-2].output
#     new_output = Dense(actions.shape[0], activation='softmax', name='output_dense')(x)
#     new_model = Model(inputs=model.input, outputs=new_output)
#
#     # Freeze all layers except the last
#     for layer in new_model.layers[:-2]:
#         layer.trainable = False
#
#     new_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#     return new_model
#
#
# # Real-time Detection
# def real_time_detection():
#     sequence, sentence, predictions = [], [], []
#     threshold = 0.5
#     cap = cv2.VideoCapture(0)
#
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             image, results = mediapipe_detection(frame, holistic)
#             draw_styled_landmarks(image, results)
#
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]
#
#             if len(sequence) == 30:
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]
#                 predictions.append(np.argmax(res))
#
#                 if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
#                     if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
#                         sentence.append(actions[np.argmax(res)])
#
#                 if len(sentence) > 5:
#                     sentence = sentence[-5:]
#
#             cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
#             cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#             cv2.imshow('OpenCV Feed', image)
#
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     collect_data()
#     X_train, X_test, y_train, y_test = preprocess_data()
#
#     model = load_and_update_model()
#     if model is None:
#         model = Sequential([
#             LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
#             LSTM(128, return_sequences=True, activation='relu'),
#             LSTM(64, return_sequences=False, activation='relu'),
#             Dense(64, activation='relu'),
#             Dense(32, activation='relu'),
#             Dense(actions.shape[0], activation='softmax')
#         ])
#         model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#
#     model.fit(X_train, y_train, epochs=200)
#     model.save('action.h5')
#     real_time_detection()




import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join('MP_Data')
existing_actions = np.array(['thanks'])
new_actions = np.array(['help'])
actions = np.concatenate([existing_actions, new_actions])

no_sequences = 25
sequence_length = 30

def setup_folders():
    for action in actions:
        for sequence in range(no_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

def collect_data():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in new_actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, f'STARTING COLLECTION: {action}', (50,200),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting {action}', (50,50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()

def preprocess_data():
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(actions.tolist().index(action))

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return train_test_split(X, y, test_size=0.05)

def load_and_update_model():
    try:
        model = load_model('action.h5')
        print("Loaded existing model.")
    except:
        print("No existing model found. Training from scratch.")
        return None

    x = model.layers[-2].output
    new_output = Dense(actions.shape[0], activation='softmax', name='output_dense')(x)
    new_model = Model(inputs=model.input, outputs=new_output)

    for layer in new_model.layers[:-2]:
        layer.trainable = False

    new_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return new_model

def real_time_detection(model):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                    if len(sentence) == 0 or actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_folders()
    collect_data()
    X_train, X_test, y_train, y_test = preprocess_data()

    model = load_and_update_model()
    if model is None:
        model = Sequential([
            LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(actions.shape[0], activation='softmax')
        ])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=200)
    model.save('action.h5')
    real_time_detection(model)