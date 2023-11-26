from flask import Flask, render_template, Response, jsonify
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import random

app = Flask(__name__)

# 모델 로드
model = load_model("keras_Model.h5", compile=False)

# UTF-8 인코딩으로 레이블 로드
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

# keras 모델에 피드할 적절한 모양의 배열 생성
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 카메라에 연결 (0은 일반적으로 기본 카메라)
cap = cv2.VideoCapture(0)

# 행동 감지를 위한 변수
action_interval = 20  # 행동을 감지할 간격 (초)
start_time = time.time()
hints_given = 0
rounds_played = 0

# 가위바위보 게임 변수
user_choice = None
computer_choice = None
result = None
game_results = []

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

# 힌트 제공 함수
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

def generate_frames():
    global model, class_names, data, cap, action_interval, start_time, hints_given, rounds_played
    global user_choice, computer_choice, result, game_results

    while True:
        ret, frame = cap.read()

        # 프레임을 PIL 이미지로 변환
        image = Image.fromarray(frame)

        # 이미지 크기 조절 및 전처리
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # 클래스 예측
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # 예측과 함께 프레임 표시
        cv2.putText(frame, f"Class: {class_name[2:]} - Confidence: {confidence_score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 남은 시간 계산 및 표시
        elapsed_time = time.time() - start_time
        remaining_time = max(0, action_interval - elapsed_time)
        cv2.putText(frame, f"Remaining Time: {remaining_time:.2f} seconds", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play_game')
def play_game():
    global user_choice, computer_choice, result, game_results

    # 가위바위보 게임 로직 추가
    if user_choice is not None and computer_choice is not None and result is not None:
        game_results.append({
            'user_choice': user_choice,
            'computer_choice': computer_choice,
            'result': result
        })

        # 결과를 HTML에 업데이트
        return jsonify({
            'user_choice': user_choice,
            'computer_choice': computer_choice,
            'result': result
        })
    else:
        return jsonify({'error': 'Game not played yet.'})

if __name__ == "__main__":
    app.run(debug=True)
