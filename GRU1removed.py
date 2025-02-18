import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import cv2


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
actions = np.array(['thanks', 'help', 'please'])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
 """Draws landmarks and connections"""
 if results.face_landmarks:
  selected_face_landmarks = [1,
 33,
 263,
 196,
 362,
 451,
 118,
 365,
 300,
 247,
 172,
 443,
 164,
 329,
 433,
 428,
 295,
 99,
 353,
 84,
 278,
 53,
 133,
 20,
 312,
 438,
 67,
 327,
 380,
 145,
 16,
 330,
 455,
 367,
 108,
 179,
 372,
 403,
 382,
 294,
 288,
 381,
 286,
 429,
 75,
 453,
 139,
 369,
 32,
 356,
 359,
 427,
 64,
 309,
 414,
 245,
 261,
 315,
 386,
 129,
 338,
 213,
 202,
 436,
 314,
 152,
 46,
 203,
 299,
 28,
 379,
 70,
 116,
 204,
 420,
 243,
 132,
 346,
 347,
 58,
 50,
 106,
 417,
 49,
 199,
 21,
 452,
 287,
 117,
 4,
 424,
 258,
 301,
 194,
 445,
 253,
 252,
 143,
 339,
 397,
 115,
 54,
 256,
 246,
 255,
 65,
 457,
 154,
 422,
 159,
 176,
 348,
 9,
 2,
 231,
 325,
 171,
 6,
 33,
 275,
 207,
 22,
 51,
 73,
 66,
 289,
 150,
 146,
 412,
 192,
 354,
 38,
 39,
 306,
 123,
 103,
 79,
 138,
 464,
 87,
 366,
 399,
 319,
 423,
 36,
 458,
 71,
 182,
 266,
 61,
 104,
 8,
 59,
 45,
 55,
 63,
 7,
 130,
 406,
 41,
 375,
 215,
 360,
 465,
 153,
 230,
 302,
 248,
 167,
 459,
 463,
 136,
 134,
 210,
 454,
 439,
 307,
 178,
 351,
 387,
 235,
 124,
 10,
 149,
 93,
 456,
 285,
 125,
 109,
 151,
 100,
 236,
 259,
 102,
 208,
 284,
 31,
 140,
 442,
 180,
 450,
 175,
 460]  # shortened for brevity
  for i in selected_face_landmarks:
   x = int(results.face_landmarks.landmark[i].x * image.shape[1])
   y = int(results.face_landmarks.landmark[i].y * image.shape[0])
   cv2.circle(image, (x, y), 1, (80, 110, 10), -1)

 if results.pose_landmarks:
  mp_drawing.draw_landmarks(
   image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
   mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
   mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
  )

 if results.left_hand_landmarks:
  mp_drawing.draw_landmarks(
   image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
   mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
   mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
  )

 if results.right_hand_landmarks:
  mp_drawing.draw_landmarks(
   image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
  )


def extract_keypoints(results):
 pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                  results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

 selected_face_landmarks = [1,
 33,
 263,
 196,
 362,
 451,
 118,
 365,
 300,
 247,
 172,
 443,
 164,
 329,
 433,
 428,
 295,
 99,
 353,
 84,
 278,
 53,
 133,
 20,
 312,
 438,
 67,
 327,
 380,
 145,
 16,
 330,
 455,
 367,
 108,
 179,
 372,
 403,
 382,
 294,
 288,
 381,
 286,
 429,
 75,
 453,
 139,
 369,
 32,
 356,
 359,
 427,
 64,
 309,
 414,
 245,
 261,
 315,
 386,
 129,
 338,
 213,
 202,
 436,
 314,
 152,
 46,
 203,
 299,
 28,
 379,
 70,
 116,
 204,
 420,
 243,
 132,
 346,
 347,
 58,
 50,
 106,
 417,
 49,
 199,
 21,
 452,
 287,
 117,
 4,
 424,
 258,
 301,
 194,
 445,
 253,
 252,
 143,
 339,
 397,
 115,
 54,
 256,
 246,
 255,
 65,
 457,
 154,
 422,
 159,
 176,
 348,
 9,
 2,
 231,
 325,
 171,
 6,
 33,
 275,
 207,
 22,
 51,
 73,
 66,
 289,
 150,
 146,
 412,
 192,
 354,
 38,
 39,
 306,
 123,
 103,
 79,
 138,
 464,
 87,
 366,
 399,
 319,
 423,
 36,
 458,
 71,
 182,
 266,
 61,
 104,
 8,
 59,
 45,
 55,
 63,
 7,
 130,
 406,
 41,
 375,
 215,
 360,
 465,
 153,
 230,
 302,
 248,
 167,
 459,
 463,
 136,
 134,
 210,
 454,
 439,
 307,
 178,
 351,
 387,
 235,
 124,
 10,
 149,
 93,
 456,
 285,
 125,
 109,
 151,
 100,
 236,
 259,
 102,
 208,
 284,
 31,
 140,
 442,
 180,
 450,
 175,
 460]  # shortened for brevity
 face = np.array([[res.x, res.y, res.z] for i, res in enumerate(results.face_landmarks.landmark)
                  if i in selected_face_landmarks]).flatten() if results.face_landmarks else np.zeros(
  len(selected_face_landmarks) * 3)

 lh = np.array([[res.x, res.y, res.z] for res in
                results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

 rh = np.array([[res.x, res.y, res.z] for res in
                results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

 return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame

def real_time_detection():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    actions = np.array(['thanks', 'help', 'please'])
    colors = [(245, 117, 16), (16, 117, 245), (16, 66, 226)]

    model = load_model('action4GRU.h5')

    # Initialize PiCamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            # Capture frame from Raspberry Pi camera
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                # Visualization logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Visualize probabilities
                image = prob_viz(res, actions, image, colors)

            # Display result
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('ASL Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        picam2.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    print("Starting real-time detection...")
    real_time_detection()
