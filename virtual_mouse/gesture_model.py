import cv2
import numpy as np
import pyautogui
from keras.models import load_model
from virtual_mouse.utils import preprocess_frame, get_prediction_label

# Load the trained model
model = load_model('../model/gesture_model.h5')

# Define the gesture labels (must match training)
labels = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']

cap = cv2.VideoCapture(0)
img_size = 256
screen_width, screen_height = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw Region of Interest (ROI)
    roi = frame[100:400, 100:400]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    processed = preprocess_frame(gray, img_size)

    prediction = model.predict(processed)
    pred_label = get_prediction_label(prediction, labels)

    # Draw prediction and ROI on original frame
    cv2.putText(frame, f"Gesture: {pred_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Define mouse actions based on gesture
    if pred_label == 'fist':
        # Map ROI center to screen position
        roi_gray = cv2.resize(gray, (img_size, img_size))
        _, thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                screen_x = int((cx / img_size) * screen_width)
                screen_y = int((cy / img_size) * screen_height)
                pyautogui.moveTo(screen_x, screen_y)

    elif pred_label == 'peace':
        pyautogui.click(button='left')

    elif pred_label == 'five':
        pyautogui.click(button='right')

    elif pred_label == 'thumbs':
        pyautogui.scroll(-50)  # Scroll down

    elif pred_label == 'rad':
        print("Sound control placeholder - implement if needed")

    # okay, straight, none => no action

    # Display
    cv2.imshow("Virtual Mouse", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
