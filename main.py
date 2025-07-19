import cv2
import numpy as np
import tensorflow as tf

# Path to the saved model directory (after extracting the downloaded model)
MODEL_PATH = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'

# Path to label map
LABEL_MAP_PATH = 'labelmap.txt'  # Should contain one label per line (COCO labels)

# Load the model
print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_PATH)
print("Model loaded successfully!")

# Load label map
with open(LABEL_MAP_PATH, 'r') as f:
    class_names = f.read().splitlines()

# Setup video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.convert_to_tensor(img)
    img = img[tf.newaxis, ...]
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    detections = detect_fn(input_tensor)

    h, w, _ = frame.shape

    boxes = detections['detection_boxes'][0].numpy()
    class_ids = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    for i in range(len(scores)):
        if scores[i] > 0.5:
            y1, x1, y2, x2 = boxes[i]
            (left, top, right, bottom) = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))

            label = class_names[class_ids[i] - 1]  # class IDs start from 1
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {int(scores[i]*100)}%', (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Real-Time Object Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
