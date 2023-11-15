import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import threading

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 원하는 너비
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 원하는 높이

# Variables for action detection
start_time = None
action_detected = False
action_threshold = 5  # The duration (in seconds) to detect the action
frame_skip = 3  # Process every 3rd frame
frame_count = 0

# Function to read frames from the camera
def read_camera():
    global frame_count
    while True:
        ret, frame = cap.read()
        if frame_count % frame_skip == 0:
            process_frame(frame)
        frame_count += 1

# Function to process each frame
def process_frame(frame):
    global action_detected, start_time
    # Convert the frame to PIL Image
    image = Image.fromarray(frame)

    # Resize and preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the frame with prediction
    cv2.putText(frame, f"Class: {class_name[2:]} - Confidence: {confidence_score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Action Detection', frame)

    # Check for action detection
    if confidence_score > 0.5:  # Adjust the confidence threshold as needed
        if not action_detected:
            action_detected = True
            start_time = time.time()
    else:
        action_detected = False

    # Check if action has been detected for the specified duration
    if action_detected and (time.time() - start_time) > action_threshold:
        print("Action Detected!")
        # Reset variables to avoid continuous detection for the same action
        action_detected = False
        start_time = None

# Start camera reading thread
camera_thread = threading.Thread(target=read_camera)
camera_thread.start()

# Main loop
while True:
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

