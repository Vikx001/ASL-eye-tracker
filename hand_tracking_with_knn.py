# hand_tracking_with_knn.py
import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import pyautogui
import os
import time

# Load the trained model
knn_model = load('asl_knn_model.joblib')

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get the class labels
asl_classes = sorted(os.listdir('asl_dataset'))


# Function to preprocess the frame for model prediction
def preprocess_frame(frame, target_size=(64, 64)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0
    frame = frame.flatten()  # Flatten the image for sklearn model
    frame = np.expand_dims(frame, axis=0)
    return frame


# Initialize webcam
video_capture = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Mode selector
mode = "control"  # Default mode is control

# String to hold recognized ASL gestures
recognized_text = ""

# Debounce variables
last_recognized_gesture = None
last_recognized_time = 0
debounce_delay = 2.0  # 2 seconds debounce delay

# Main loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    recognized_gesture = None  # Define recognized_gesture at the start of each frame processing

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around the hand
            x_min, y_min = frame_width, frame_height
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y

            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if mode == "sign_language" and hand_img.size != 0:
                # Preprocess the hand region
                preprocessed_frame = preprocess_frame(hand_img)

                # Predict the gesture
                prediction = knn_model.predict(preprocessed_frame)
                predicted_class = prediction[0]
                recognized_gesture = asl_classes[predicted_class]

                # Append recognized gesture to the text if different from last recognized gesture and after debounce delay
                current_time = time.time()
                if recognized_gesture != last_recognized_gesture or (
                        current_time - last_recognized_time) > debounce_delay:
                    recognized_text += recognized_gesture
                    last_recognized_gesture = recognized_gesture
                    last_recognized_time = current_time

    # Display recognized ASL gesture and the full text on the frame
    if recognized_gesture:
        cv2.putText(frame, f"Gesture: {recognized_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    cv2.putText(frame, f"Text: {recognized_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Display the current mode
    cv2.putText(frame, f"Mode: {mode}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Tracking", frame)

    # Switch modes with key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        mode = "control"
    elif key == ord('s'):
        mode = "sign_language"
    elif key == ord('x'):
        recognized_text = ""  # Clear the recognized text

video_capture.release()
cv2.destroyAllWindows()
