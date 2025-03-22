import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np
import screen_brightness_control as sbc  # For brightness control

# Function to detect gestures, including pinch and brightness control gestures
def detect_gesture(lst):
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2
    fingers = []

    # Thumb detection
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        fingers.append(1)  # Thumb is up
    else:
        fingers.append(0)

    # Other fingers detection
    fingers.append(1 if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh else 0)  # Index
    fingers.append(1 if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh else 0)  # Middle
    fingers.append(1 if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh else 0)  # Ring
    fingers.append(1 if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh else 0)  # Pinky

    # Gesture detection logic
    if fingers == [1, 1, 1, 1, 1]:
        return "palm"  # All fingers extended
    elif fingers == [0, 0, 0, 0, 0]:
        return "fist"  # Fist
    elif fingers == [0, 1, 0, 0, 0]:
        return "index_left" if lst.landmark[8].x * 100 < lst.landmark[7].x * 100 else "index_right"
    elif fingers == [0, 1, 1, 0, 0]:  # Two fingers up (index and middle)
        return "two_fingers_left" if lst.landmark[8].x * 100 < lst.landmark[7].x * 100 else "two_fingers_right"

    # Detect pinch gesture (thumb and index finger close)
    thumb_tip = lst.landmark[4]
    index_tip = lst.landmark[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

    if distance < 0.05:  # Adjust threshold as needed
        return "pinch"

    return "none"  # No recognizable gesture


# Initialize camera and Mediapipe
cap = cv2.VideoCapture(0)
drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=2)

prev_gesture = None
mute_triggered = False  # Track mute state to prevent rapid toggling

# Main loop for gesture recognition
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        continue

    frame = cv2.flip(frame, 1)  # Mirror the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_obj.process(frame_rgb)

    gestures = []  # Store gestures for both hands
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
            gesture = detect_gesture(hand_landmarks)
            gestures.append((gesture, hand_label))

            # Draw hand landmarks
            drawing.draw_landmarks(frame, hand_landmarks, hands.HAND_CONNECTIONS)

        # Handle mute when both hands make a fist
        if len(gestures) == 2 and all(g[0] == "fist" for g in gestures):
            if not mute_triggered:  # Trigger mute only once
                print("Muting Volume")
                pyautogui.press("volumemute")
                mute_triggered = True  # Prevent rapid toggling
        else:
            mute_triggered = False  # Reset mute trigger if fists are released

        # Handle other gestures for brightness control and media playback
        for gesture, hand_label in gestures:
            if gesture == "two_fingers_right":
                sbc.set_brightness(min(sbc.get_brightness()[0] + 10, 100))
                print("Increasing Brightness")
                time.sleep(0.5)  # Control brightness change speed

            elif gesture == "two_fingers_left":
                sbc.set_brightness(max(sbc.get_brightness()[0] - 10, 0))
                print("Decreasing Brightness")
                time.sleep(0.5)  # Control brightness change speed

            elif gesture != prev_gesture:  # Handle other gestures only on change
                if gesture == "palm":
                    pyautogui.press("space")  # Play/pause
                elif gesture == "index_right":
                    pyautogui.press("right")  # Skip forward
                elif gesture == "index_left":
                    pyautogui.press("left")  # Rewind
                elif gesture == "pinch":
                    if hand_label == "Right":
                        pyautogui.press("volumeup")  # Volume up
                    else:
                        pyautogui.press("volumedown")  # Volume down

                prev_gesture = gesture  # Update previous gesture

    # Display the frame
    cv2.imshow("Gesture Control", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
