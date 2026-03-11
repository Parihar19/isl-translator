import cv2
import mediapipe as mp
import time
import pickle
import pyttsx3
import threading 
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- THE FIX: The Traffic Light ---
is_speaking = False

def speak_letter(letter):
    global is_speaking
    is_speaking = True # Turn light RED
    try:
        # Create a fresh engine for this specific speech request (Windows loves this)
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(letter)
        engine.runAndWait()
    except Exception:
        # If Windows throws a tiny internal fit, just ignore it silently
        pass 
    finally:
        is_speaking = False # Turn light GREEN again when finished talking

# -----------------------------------

print("Loading AI Model...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, 
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7) 

MY_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       
    (0, 5), (5, 6), (6, 7), (7, 8),       
    (5, 9), (9, 10), (10, 11), (11, 12),  
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

cap = cv2.VideoCapture(0)

last_predicted = ""
frames_stable = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, c = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_data = {'Left': [0.0] * 42, 'Right': [0.0] * 42}
        predicted_letter = "?"

        if result.hand_landmarks and result.handedness:
            for idx, hand_landmarks in enumerate(result.hand_landmarks):
                hand_label = result.handedness[idx][0].category_name
                
                for connection in MY_HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_point = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    end_point = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2) 
                
                coords = []
                for landmark in hand_landmarks:
                    coords.extend([landmark.x, landmark.y])
                
                hand_data[hand_label] = coords

            combined_row_data = hand_data['Left'] + hand_data['Right']
            prediction = model.predict([combined_row_data])
            predicted_letter = prediction[0]

            if predicted_letter == last_predicted:
                frames_stable += 1
            else:
                frames_stable = 0 
                last_predicted = predicted_letter

            # FIXED: Only trigger speech if stable AND the traffic light is green!
            if frames_stable >= 15 and not is_speaking:
                threading.Thread(target=speak_letter, args=(predicted_letter,), daemon=True).start()
                frames_stable = -15 # Cooldown

        cv2.rectangle(frame, (0, 0), (250, 80), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"Sign: {predicted_letter}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        cv2.imshow('Live ISL Translator', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()