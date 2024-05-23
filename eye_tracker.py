import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Predefined simplified ASL gestures patternsc
asl_gestures = {
    'A': [0, 0, 0, 0, 0],  # Fist: all fingers folded
    'B': [0, 1, 1, 1, 1],  # Palm open: all fingers extended, thumb folded
    'C': [0, 1, 1, 1, 1],  # 'C' shape: all fingers curved (simplified as open palm)
    'D': [0, 1, 0, 0, 0],  # 'D' shape: index finger up, others folded
    'E': [0, 0, 0, 0, 0],  # 'E' shape: similar to fist (simplified)
    'F': [0, 1, 1, 0, 0],  # 'F' shape: thumb and index form circle, others extended (simplified)
    'G': [0, 1, 0, 0, 0],  # 'G' shape: index and thumb extended (simplified as D)
    'H': [0, 1, 1, 0, 0],  # 'H' shape: index and middle extended
    'I': [0, 0, 0, 0, 1],  # 'I' shape: pinky extended, others folded
    'J': [0, 0, 0, 0, 1],  # 'J' shape: similar to I, with movement (simplified)
    'K': [0, 1, 1, 0, 0],  # 'K' shape: similar to H (simplified)
    'L': [0, 1, 0, 0, 0],  # 'L' shape: index and thumb extended
    'M': [0, 0, 0, 1, 1],  # 'M' shape: three fingers over thumb
    'N': [0, 0, 1, 1, 1],  # 'N' shape: two fingers over thumb
    'O': [0, 1, 1, 1, 1],  # 'O' shape: all fingers form a circle
    'P': [0, 1, 1, 0, 0],  # 'P' shape: similar to K
    'Q': [0, 1, 0, 0, 0],  # 'Q' shape: similar to G (simplified)
    'R': [0, 1, 1, 0, 0],  # 'R' shape: index and middle crossed (simplified)
    'S': [0, 0, 0, 0, 0],  # 'S' shape: similar to fist
    'T': [0, 0, 0, 0, 0],  # 'T' shape: thumb between index and middle (simplified as fist)
    'U': [0, 1, 1, 0, 0],  # 'U' shape: index and middle extended
    'V': [0, 1, 1, 0, 0],  # 'V' shape: similar to U
    'W': [0, 1, 1, 1, 0],  # 'W' shape: three fingers extended
    'X': [0, 1, 0, 0, 0],  # 'X' shape: index finger bent (simplified)
    'Y': [1, 0, 0, 0, 1],  # 'Y' shape: thumb and pinky extended
    'Z': [0, 1, 0, 0, 0],  # 'Z' shape: index finger traces Z (simplified)
}



def map_to_screen(x, y, frame_width, frame_height, screen_width, screen_height):
    screen_x = np.interp(x, [0, frame_width], [0, screen_width])
    screen_y = np.interp(y, [0, frame_height], [0, screen_height])
    return screen_x, screen_y



def detect_click_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y]))
    return distance < 0.05


#
def detect_right_click_gesture(hand_landmarks):
    # Check if all finger tips are folded below the middle finger PIP joint
    folded = all(hand_landmarks.landmark[mp_hands.HandLandmark(i)].y > hand_landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y for i in range(5, 21))
    return folded


# Function to detect ASL gestures
def detect_asl_gesture(hand_landmarks):
    fingers = []
    tips_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    for tip_id in tips_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    for gesture, pattern in asl_gestures.items():
        if fingers == pattern:
            return gesture
    return None


# Initialize webcam
video_capture = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Debounce timers
last_left_click_time = time.time()
last_right_click_time = time.time()
click_delay = 0.5  # 500 ms delay between clicks

# String to hold recognized ASL gestures
recognized_text = ""

# Mode selector
mode = "control"  # Default mode is control

# Main loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    recognized_gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Map the hand coordinates to screen coordinates
            screen_x, screen_y = map_to_screen(index_finger_tip.x * frame_width, index_finger_tip.y * frame_height,
                                               frame_width, frame_height, screen_width, screen_height)

            if mode == "control":
                # Move the mouse cursor to the mapped coordinates
                pyautogui.moveTo(screen_x, screen_y)

                # Detect click gestures with debouncing
                current_time = time.time()
                if detect_click_gesture(hand_landmarks) and (current_time - last_left_click_time) > click_delay:
                    pyautogui.click()
                    last_left_click_time = current_time
                elif detect_right_click_gesture(hand_landmarks) and (
                        current_time - last_right_click_time) > click_delay:
                    pyautogui.rightClick()
                    last_right_click_time = current_time

            elif mode == "sign_language":
                # Detect ASL gesture
                recognized_gesture = detect_asl_gesture(hand_landmarks)

                # Append recognized gesture to the text
                if recognized_gesture:
                    recognized_text += recognized_gesture

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

video_capture.release()
cv2.destroyAllWindows()
