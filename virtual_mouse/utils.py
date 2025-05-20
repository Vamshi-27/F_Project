import numpy as np

def preprocess_frame(frame, img_size):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame / 255.0
    frame = frame.reshape(1, img_size, img_size, 1)
    return frame

def get_prediction_label(prediction, labels):
    return labels[np.argmax(prediction)]