import cv2
import mediapipe as mp
import time
import math # NEW: Imported for distance calculation

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, 
    num_hands=2)

MY_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       
    (0, 5), (5, 6), (6, 7), (7, 8),       
    (5, 9), (9, 10), (10, 11), (11, 12),  
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, c = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                
                # Draw the web
                for connection in MY_HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    start_point = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    end_point = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2) 
                
                for landmark in hand_landmarks:
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 

                # --- NEW GESTURE RECOGNITION LOGIC ---
                
                # Get coordinates for Thumb Tip (4) and Index Tip (8)
                thumb_tip_x = int(hand_landmarks[4].x * w)
                thumb_tip_y = int(hand_landmarks[4].y * h)
                
                index_tip_x = int(hand_landmarks[8].x * w)
                index_tip_y = int(hand_landmarks[8].y * h)
                
                # Calculate the Euclidean distance between the two points
                distance = math.hypot(index_tip_x - thumb_tip_x, index_tip_y - thumb_tip_y)
                
                # If the distance is less than 40 pixels, consider it a "Pinch"
                if distance < 40:
                    # Draw a green circle between the fingers to show it detected the pinch
                    cv2.circle(frame, ((thumb_tip_x + index_tip_x) // 2, (thumb_tip_y + index_tip_y) // 2), 15, (0, 255, 0), cv2.FILLED)
                    
                    # Write text on the screen!
                    cv2.putText(frame, "PINCH DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow('ISL Translator - Gesture Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()