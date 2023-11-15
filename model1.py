import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import random

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels with UTF-8 encoding
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Open a connection to the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Variables for action detection
start_time = time.time()
action_interval = 5  # The interval (in seconds) to detect the action
action_detected = False

# Rock-paper-scissors game variables
user_choice = None
computer_choice = None
result = None

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

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
    cv2.imshow('Rock-Paper-Scissors Game', frame)

    # Check for action detection
    elapsed_time = time.time() - start_time
    remaining_time = action_interval - elapsed_time

    if confidence_score > 0.5 and elapsed_time >= action_interval:
        print(f"Action Detected! Time: {elapsed_time:.2f} seconds")
        
        # Map class names to rock, paper, or scissors
        if "Rock" in class_name:
            user_choice = "Rock"
        elif "Paper" in class_name:
            user_choice = "Paper"
        elif "Scissors" in class_name:
            user_choice = "Scissors"

        # Get computer's choice
        computer_choice = get_computer_choice()

        # Determine the game result
        if user_choice == computer_choice:
            result = "It's a tie!"
        elif (
            (user_choice == "Rock" and computer_choice == "Scissors") or
            (user_choice == "Paper" and computer_choice == "Rock") or
            (user_choice == "Scissors" and computer_choice == "Paper")
        ):
            result = "You win!"
        else:
            result = "You lose!"

        # Print and reset variables for the next detection
        print(f"User Choice: {user_choice}, Computer Choice: {computer_choice}")
        print(result)
        action_detected = True
        start_time = time.time()

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


