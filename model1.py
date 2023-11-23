from flask import Flask, render_template, request, redirect, url_for
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import random

app = Flask(__name__)

# 모델 로드 및 기타 초기화 코드
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
        return hints[hints_given]
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game_start')
def game_start():
    hint = provide_hint()
    return render_template('game_start.html', hint=hint)

@app.route('/game')
def game():
    global cap, start_time, hints_given, rounds_played, user_choice, computer_choice, result
    
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
    
    elapsed_time = time.time() - start_time
    remaining_time = max(0, action_interval - elapsed_time)
    
    if confidence_score > 0.5 and elapsed_time >= action_interval:
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
            result = "이겼습니다! " + provide_hint()
        else:
            result = "이런~ 지셨군요. 다시 도전해 보세요!"
        
        action_detected = True
        start_time = time.time()
        rounds_played += 1
        
        if rounds_played == 5:
            return redirect(url_for('game_over'))
    
    return render_template('game.html', frame=frame, class_name=class_name[2:], confidence_score=confidence_score, remaining_time=remaining_time, result=result, user_choice=user_choice, computer_choice=computer_choice)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/guess', methods=['POST'])
def guess():
    user_guess = request.form['user_guess']
    if user_guess == "김연아":
        return redirect(url_for('game_result', result="정답입니다! 축하합니다!"))
    else:
        return redirect(url_for('game_result', result="아쉽군요. 다시 도전해 보세요!"))

@app.route('/more_hints')
def more_hints():
    hint = provide_hint()
    return render_template('more_hints.html', hint=hint)

@app.route('/game_records')
def game_records():
    # 여기에 게임 결과를 기록하고 보여주는 로직 추가
    records = ["Game 1: Win", "Game 2: Draw", "Game 3: Loss"]
    return render_template('game_records.html', records=records)

@app.route('/game_over', methods=['GET', 'POST'])
def game_over():
    global rounds_played
    if request.method == 'POST':
        user_guess = request.form['user_guess']
        if user_guess == "김연아":
            return redirect(url_for('game_result', result="정답입니다! 축하합니다!"))
        else:
            return redirect(url_for('game_result', result="아쉽군요. 다시 도전해 보세요!"))
    return render_template('game_over.html')

@app.route('/game_result/<result>')
def game_result(result):
    return render_template('game_result.html', result=result, rounds_played=rounds_played)

if __name__ == '__main__':
    app.run(debug=True)
