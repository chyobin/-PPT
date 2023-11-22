from flask import Flask, render_template, request, redirect
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import random

app = Flask(__name__)

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r", encoding="utf-8").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

cap = cv2.VideoCapture(0)
action_interval = 20
start_time = time.time()
hints_given = 0
rounds_played = 0

user_choice = None
computer_choice = None
result = None

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def provide_hint():
    global hints_given
    hints = [
        "이 인물은 스포츠 선수입니다.",
        "이 인물은 올림픽 메달리스트입니다.",
        "이 인물은 한국의 유명한 가수이기도 합니다.",
        "이 인물은 배우로서도 활동하고 있습니다.",
        "이 인물은 대중 매체에 자주 등장하는 연예인입니다."
    ]
    if hints_given < len(hints):
        print("힌트: " + hints[hints_given])
        hints_given += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game', methods=['POST'])
def game():
    global user_choice, computer_choice, result, rounds_played  # 이 줄을 추가해주세요
    ret, frame = cap.read()
    image = Image.fromarray(frame)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    cv2.putText(frame, f"Class: {class_name[2:]} - Confidence: {confidence_score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elapsed_time = time.time() - start_time
    remaining_time = max(0, action_interval - elapsed_time)
    cv2.putText(frame, f"Remaining Time: {remaining_time:.2f} seconds", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Rock-Paper-Scissors Game', frame)

    if confidence_score > 0.5 and elapsed_time >= action_interval:
        print(f"Action Detected! Time: {elapsed_time:.2f} seconds")

        if "Rock" in class_name:
            user_choice = "Rock"
        elif "Paper" in class_name:
            user_choice = "Paper"
        elif "Scissors" in class_name:
            user_choice = "Scissors"

        computer_choice = get_computer_choice()

        if user_choice == computer_choice:
            result = "무승부입니다!"
        elif (
            (user_choice == "Rock" and computer_choice == "Scissors") or
            (user_choice == "Paper" and computer_choice == "Rock") or
            (user_choice == "Scissors" and computer_choice == "Paper")
        ):
            result = "이겼습니다! "
            provide_hint()
        else:
            result = "이런~ 지셨군요. 다시 도전해 보세요!"

        print(f"User Choice: {user_choice}, Computer Choice: {computer_choice}")
        print(result)
        hints_given = 0
        rounds_played += 1

        if rounds_played == 5:
            print("5판이 끝났습니다. 이제 인물을 맞춰보세요!")
            return render_template('guess.html')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return redirect('/')  # 'q'를 누르면 index.html로 리디렉션합니다.

    return render_template('game.html', user_choice=user_choice, computer_choice=computer_choice, result=result)

@app.route('/result', methods=['POST'])
def result():
    global user_guess, correct
    print("기회가 끝났습니다. 이 사람은 누굴까요?")
    user_guess = request.form.get("user_guess")

    if user_guess == "김연아":
        correct = True
    else:
        correct = False

    return render_template('result.html', user_guess=user_guess, correct=correct)

if __name__ == '__main__':
    app.run(debug=True)
