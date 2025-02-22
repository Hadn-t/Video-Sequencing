def mediapipe_detection(image, model):
    # Reduce image resolution for faster processing
    image = cv2.resize(image, (320, 240))  # Half resolution
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results is None:
        return image
    
    # Simplified landmark drawing with reduced thickness
    if results.face_landmarks:
        for i in selected_face_landmarks:
            try:
                x = int(results.face_landmarks.landmark[i].x * image.shape[1])
                y = int(results.face_landmarks.landmark[i].y * image.shape[0])
                cv2.circle(image, (x, y), 1, (80, 110, 10), 1)  # Reduced size
            except IndexError:
                continue

    # Simplified drawing specs for better performance
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            drawing_spec, drawing_spec)

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            drawing_spec, drawing_spec)

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            drawing_spec, drawing_spec)

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

    return np.concatenate([pose, face[:len(selected_face_landmarks) * 3], lh, rh])[:864]

def prob_viz(res, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

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
        print(f"Text To Speech Error: {str(e)}")


def generate_sentence(words):
    # First, clean and deduplicate words while preserving order
    word_list = words.strip().split()
    seen = set()
    unique_words = []
    for word in word_list:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)

    words = " ".join(unique_words)

    prompt = f'''Create a meaningful sentence using these words: "{words}"
    Rules:
    1. If "namaste" is present, start the sentence with "Namaste" as a greeting
    2. After greeting, use emergency words (help, danger, emergency) next if present
    3. Use the remaining words to complete the sentence
    4. Remove duplicate words
    5. Add only basic connectors (I, am, in, need, have, feel, please, the, is) if needed
    6. Create a clear, logical sentence that makes sense in an emergency context
    7. End with appropriate punctuation (! for emergency words, . for normal statements)
    8. Maximum one exclamation mark per sentence


    Make sure the sentence flows naturally and makes sense in an emergency context.'''

    # Example input: "danger thanks danger pain namaste"
    # Example output: "Namaste, I am in danger and feel pain, please help!"

    try:
        response = requests.post(
            GEMINI_API_URL,
            json={
                "contents": [{"parts": [{"text": prompt}]}]
            }
        )
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"Error: {str(e)}"




if __name__ == "__main__":
    # Define actions and colors
    actions = np.array(['thanks', 'help', 'please', 'danger', 'pain', 'namaste', 'donate', 'Mother', 'father', 'Nice_to_meet_you', 'Learning'])
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (32, 117, 16),
              (64, 117, 16), (88, 117, 16), (60, 117, 16),(60, 117, 16),(60, 117, 16),(60, 117, 16),(60, 117, 16)]

    # Selected face landmarks to reduce the delay by reducing the points to rem
    selected_face_landmarks = [1, 33, 263, 196, 362, 451, 118, 365, 300, 247, 172, 443,
                             164, 329, 433, 428, 295, 99, 353, 84, 278, 53, 133, 20,
                             312, 438, 67, 327, 380, 145, 16, 330, 455, 367, 108, 179,
                             372, 403, 382, 294, 288, 381, 286, 429, 75, 453, 139, 369,
                             32, 356, 359, 427, 64, 309, 414, 245, 261, 315, 386, 129,
                             338, 213, 202, 436, 314, 152, 46, 203, 299, 28, 379, 70,
                             116, 204, 420, 243, 132, 346, 347, 58, 50, 106, 417, 49,
                             199, 21, 452, 287, 117, 4, 424, 258, 301, 194, 445, 253,
                             252, 143, 339, 397, 115, 54, 256, 246, 255, 65, 457, 154,
                             422, 159, 176, 348, 9, 2, 231, 325, 171, 6, 33, 275, 207,
                             22, 51, 73, 66, 289, 150, 146, 412, 192, 354, 38, 39, 306,
                             123, 103, 79, 138, 464, 87, 366, 399, 319, 423, 36, 458,
                             71, 182, 266, 61, 104, 8, 59, 45, 55, 63, 7, 130, 406, 41,
                             375, 215, 360, 465, 153, 230, 302, 248, 167, 459, 463, 136,
                             134, 210, 454, 439, 307, 178, 351, 387, 235, 124, 10, 149,
                             93, 456, 285, 125, 109, 151, 100, 236, 259, 102, 208, 284,
                             31, 140, 442, 180, 450, 175, 460]

    print("Starting real-time ASL detection...")
    real_time_detection()
