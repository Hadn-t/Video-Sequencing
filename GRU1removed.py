import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import cv2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results is None:
        return image
        
    if results.face_landmarks:
        for i in selected_face_landmarks:
            try:
                x = int(results.face_landmarks.landmark[i].x * image.shape[1])
                y = int(results.face_landmarks.landmark[i].y * image.shape[0])
                cv2.circle(image, (x, y), 1, (80, 110, 10), -1)
            except IndexError:
                continue

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
    # Initialize arrays for each component
    pose = np.zeros(33 * 4)
    face = np.zeros(len(selected_face_landmarks) * 3)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    # Extract pose features if available
    if results and results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in 
                        results.pose_landmarks.landmark]).flatten()
    
    # Extract face features if available
    if results and results.face_landmarks:
        face_coords = []
        for i in selected_face_landmarks:
            try:
                landmark = results.face_landmarks.landmark[i]
                face_coords.extend([landmark.x, landmark.y, landmark.z])
            except IndexError:
                face_coords.extend([0.0, 0.0, 0.0])
        face = np.array(face_coords)
    
    # Extract hand features if available
    if results and results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in 
                      results.left_hand_landmarks.landmark]).flatten()
    
    if results and results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in 
                      results.right_hand_landmarks.landmark]).flatten()
    
    # Ensure we're returning exactly 864 features
    return np.concatenate([pose, face[:len(selected_face_landmarks)*3], lh, rh])[:864]

def prob_viz(res, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
    return output_frame

def real_time_detection():
    try:
        # Load model first to ensure it's available
        model = load_model('action4GRU.h5')

        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        # Initialize camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                try:
                    frame = picam2.capture_array()
                    if frame is None:
                        print("Failed to capture frame")
                        continue
                        
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Verify results before processing
                    if results is None:
                        print("No detection results")
                        continue
                        
                    draw_landmarks(image, results)

                    # Prediction logic
                    try:
                        keypoints = extract_keypoints(results)
                        if keypoints is None or len(keypoints) != 864:
                            print(f"Invalid keypoints length: {len(keypoints) if keypoints is not None else 'None'}")
                            continue
                            
                        sequence.append(keypoints)
                    except Exception as e:
                        print(f"Keypoint extraction error: {str(e)}")
                        continue

                    # Keep only last 30 frames
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        try:
                            # Convert sequence to numpy array and ensure correct shape
                            input_data = np.array(sequence)
                            if input_data.shape != (30, 864):
                                print(f"Invalid input shape: {input_data.shape}")
                                continue

                            res = model.predict(np.expand_dims(input_data, axis=0))[0]
                            
                            if len(res) != len(actions):
                                print(f"Prediction shape mismatch: {len(res)} vs {len(actions)}")
                                continue
                                
                            predictions.append(np.argmax(res))

                            if len(predictions) >= 10:
                                recent_pred = predictions[-10:]
                                if len(np.unique(recent_pred)) == 1 and res[np.argmax(res)] > threshold:
                                    action = actions[np.argmax(res)]
                                    if not sentence or action != sentence[-1]:
                                        sentence.append(action)

                            # Keep sentence at reasonable length
                            sentence = sentence[-5:]

                            image = prob_viz(res, image, colors)

                        except Exception as e:
                            print(f"Prediction error: {str(e)}")
                            continue

                    # Draw the sentence
                    try:
                        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, ' '.join(sentence), (3, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Drawing error: {str(e)}")
                        continue

                    cv2.imshow('ASL Detection', image)

                except Exception as e:
                    print(f"Frame processing error: {str(e)}")
                    continue

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # Keep your original variables
    actions = np.array(['thanks', 'help', 'please'])
    colors = [(245, 117, 16), (117, 245, 16), (16, 66, 226)]
    
    # Your original selected_face_landmarks
    selected_face_landmarks = [1, 33, 263, 196, 362, 451, 118, 365, 300, 247, 172, 443, 164, 329, 433, 428, 295, 99, 353, 84, 
                             278, 53, 133, 20, 312, 438, 67, 327, 380, 145, 16, 330, 455, 367, 108, 179, 372, 403, 382, 294,
                             288, 381, 286, 429, 75, 453, 139, 369, 32, 356, 359, 427, 64, 309, 414, 245, 261, 315, 386, 129,
                             338, 213, 202, 436, 314, 152, 46, 203, 299, 28, 379, 70, 116, 204, 420, 243, 132, 346, 347, 58,
                             50, 106, 417, 49, 199, 21, 452, 287, 117, 4, 424, 258, 301, 194, 445, 253, 252, 143, 339, 397,
                             115, 54, 256, 246, 255, 65, 457, 154, 422, 159, 176, 348, 9, 2, 231, 325, 171, 6, 33, 275, 207,
                             22, 51, 73, 66, 289, 150, 146, 412, 192, 354, 38, 39, 306, 123, 103, 79, 138, 464, 87, 366, 399,
                             319, 423, 36, 458, 71, 182, 266, 61, 104, 8, 59, 45, 55, 63, 7, 130, 406, 41, 375, 215, 360,
                             465, 153, 230, 302, 248, 167, 459, 463, 136, 134, 210, 454, 439, 307, 178, 351, 387, 235, 124,
                             10, 149, 93, 456, 285, 125, 109, 151, 100, 236, 259, 102, 208, 284, 31, 140, 442, 180, 450,
                             175, 460]
    
    print("Starting real-time detection...")
    real_time_detection()
