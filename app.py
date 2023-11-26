from flask import Flask, render_template, Response, request
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
        return "힌트: " + hints[hints_given]
        hints_given += 1
    return ""

def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

def gen_frames():
    global cap, model, data, class_names, action_interval, start_time, hints_given, rounds_played
    global user_choice, computer_choice, result

    while True:
        # 프레임 단위로 캡처
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

        # 행동 감지 확인
        if confidence_score > 0.5 and elapsed_time >= action_interval:
            print(f"Action Detected! Time: {elapsed_time:.2f} seconds")

            # 클래스 이름을 가위, 바위 또는 보로 매핑
            if "Rock" in class_name:
                user_choice = "Rock"
            elif "Paper" in class_name:
                user_choice = "Paper"
            elif "Scissors" in class_name:
                user_choice = "Scissors"

            # 컴퓨터의 선택 얻기
            computer_choice = get_computer_choice()

            # 게임 결과 결정
            if user_choice == computer_choice:
                result = "무승부입니다!"
            elif (
                (user_choice == "Rock" and computer_choice == "Scissors") or
                (user_choice == "Paper" and computer_choice == "Rock") or
                (user_choice == "Scissors" and computer_choice == "Paper")
            ):
                result = "이겼습니다! " + provide_hint()  # 이기면 힌트 제공
            else:
                result = "이런~ 지셨군요. 다시 도전해 보세요!"

            # 결과 출력 및 다음 감지를 위한 변수 재설정
            print(f"User Choice: {user_choice}, Computer Choice: {computer_choice}")
            print(result)
            action_detected = True
            start_time = time.time()
            hints_given = 0
            rounds_played += 1

            # 5판을 다 플레이했는지 확인
            if rounds_played == 5:
                print("5판이 끝났습니다. 이제 인물을 맞춰보세요!")
                break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global result
    return render_template('index.html', result=result)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hint')
def hint():
    global user_choice, computer_choice, result
    if result and "이겼습니다!" in result:
        return provide_hint()
    return ""

@app.route('/guess', methods=['POST'])
def guess():
    global user_choice, computer_choice, result
    user_guess = request.form.get('user_guess')
    
    # 사용자의 추측이 정확한지 확인
    if user_guess == "김연아":
        result = "정답입니다! 축하합니다!"
    else:
        result = "아쉽군요. 다시 도전해 보세요!"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)