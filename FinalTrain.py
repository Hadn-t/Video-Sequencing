
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from picamera2 import Picamera2
import cv2

# Screen Configurations
import requests
import pyttsx3
import time
from RPLCD import CharLCD
from RPLCD.i2c import CharLCD

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=20, rows=4, dotsize=8)

GOOGLE_GEMINI_API_KEY = "AIzaSyDXI81MAhaf67v8YTo5CdMS1yC15LNgFWE"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GOOGLE_GEMINI_API_KEY}"

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
    return np.concatenate([pose, face[:len(selected_face_landmarks) * 3], lh, rh])[:864]


def prob_viz(res, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
    return output_frame


def real_time_detection():
    try:
        # Loading the model first to ensure it's available
        model = load_model('action4GRU.h5')

        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        # Clearing the file before writing onto it
        with open('recognized_words.txt', 'w') as file:
            file.write('') # Clears the file

        # Initializing the camera
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()

        start_time = time.time()
        duration = 180

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while (time.time() - start_time) < duration:
                try:
                    frame = picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    # Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)

                    # Keep only last 30 frames
                    if len(sequence) > 30:
                        sequence = sequence[-30:]

                    if len(sequence) == 30:
                        # Convert sequence to numpy array and ensure correct shape
                        input_data = np.array(sequence)
                        if input_data.shape != (30, 864):
                            continue

                        res = model.predict(np.expand_dims(input_data, axis=0))[0]
                        predictions.append(np.argmax(res))

                        if np.unique(predictions[-10:])[0] == np.argmax(res):
                            if res[np.argmax(res)] > threshold:
                                if len(sentence) > 0:
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        image = prob_viz(res, image, colors)

                    remaining_time = int(duration-(time.time() - start_time))
                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, f'{" ".join(sentence)} | Time left: {remaining_time}s', (7, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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

def setup_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    return engine

def speak_text(engine, text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {str(e)}")

def generate_sentence(words):
    prompt = f'''Create a simple sentence using these words: "{words}"
    Rules:
    1. Use emergency words (help, danger, emergency) first if present
    2. Use all given words as-is
    3. Add only basic connectors (a, an, the, is) if needed
    4. Make one short, clear sentence
    5. End with a period'''

    try:
        response = requests.post(
            GEMINI_API_URL,
            json={
                "contents": [{"parts": [{"text": prompt}]}]
            }
        )
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error: {str(e)}"


def display_on_lcd(text):
    try:
        # Clear the LCD
        lcd.clear()

        # Calculate total width needed for text with padding
        padded_text = "    " + text + "    "  # Adding padding for smooth scroll

        # Scroll text horizontally on rows 2 and 3 only
        start_time = time.time()
        display_duration = 10  # Display for 10 seconds

        while time.time() - start_time < display_duration:
            for i in range(len(padded_text)):
                # Clear only rows 2 and 3
                lcd.cursor_pos = (1, 0)
                lcd.write_string(" " * 20)

                # Calculate text segment to display (20 characters)
                text_segment = padded_text[i:i + 20]
                if len(text_segment) < 20:
                    # Wrap around to the beginning of text
                    text_segment += padded_text[:20 - len(text_segment)]

                # Display the same text on rows 2 and 3
                lcd.cursor_pos = (1, 0)
                lcd.write_string(text_segment)

                # Control scroll speed
                time.sleep(0.3)  # Adjust this value to control scroll speed

                # Check if display duration has elapsed
                if time.time() - start_time >= display_duration:
                    break

    except Exception as e:
        print(f"LCD Error: {str(e)}")
    finally:
        # Clearing the LCD when done
        lcd.clear()

def screen_sound():
    try:
        # Initialize text to speech
        tts_engine = setup_tts()

        # Getting the input from the transcipted file
        with open('recognized_words.txt', 'r') as file:
            words = file.read().strip()

            if not words:
                print('No words recognized.')
                return
        # Generating the improvised prompt
        sentence = generate_sentence(words)

        # Printing to the console
        print("\n Generated Sentence:", "-" * 24)
        print(sentence)
        print("-" * 24)

        # Displaying the output on the LCD
        display_on_lcd(sentence)

        # Output the text on through the speaker
        speak_text(tts_engine, sentence)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clear the LCD when done
        lcd.clear()

if __name__ == "__main__":
    # Keep your original variables
    actions = np.array(['thanks', 'help', 'please', 'danger', 'pain', 'namaste', 'donate'])
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (32, 117, 16), (64, 117, 16), (88, 117, 16), (60, 117, 16)]

    # Your original selected_face_landmarks
    selected_face_landmarks = [1, 33, 263, 196, 362, 451, 118, 365, 300, 247, 172, 443, 164, 329, 433, 428, 295, 99,
                               353, 84,
                               278, 53, 133, 20, 312, 438, 67, 327, 380, 145, 16, 330, 455, 367, 108, 179, 372, 403,
                               382, 294,
                               288, 381, 286, 429, 75, 453, 139, 369, 32, 356, 359, 427, 64, 309, 414, 245, 261, 315,
                               386, 129,
                               338, 213, 202, 436, 314, 152, 46, 203, 299, 28, 379, 70, 116, 204, 420, 243, 132, 346,
                               347, 58,
                               50, 106, 417, 49, 199, 21, 452, 287, 117, 4, 424, 258, 301, 194, 445, 253, 252, 143, 339,
                               397,
                               115, 54, 256, 246, 255, 65, 457, 154, 422, 159, 176, 348, 9, 2, 231, 325, 171, 6, 33,
                               275, 207,
                               22, 51, 73, 66, 289, 150, 146, 412, 192, 354, 38, 39, 306, 123, 103, 79, 138, 464, 87,
                               366, 399,
                               319, 423, 36, 458, 71, 182, 266, 61, 104, 8, 59, 45, 55, 63, 7, 130, 406, 41, 375, 215,
                               360,
                               465, 153, 230, 302, 248, 167, 459, 463, 136, 134, 210, 454, 439, 307, 178, 351, 387, 235,
                               124,
                               10, 149, 93, 456, 285, 125, 109, 151, 100, 236, 259, 102, 208, 284, 31, 140, 442, 180,
                               450,
                               175, 460]

    print("Starting real-time detection...")
    real_time_detection()

    # After sucessfull transciption the text into transcipted.txt file
    screen_sound()
