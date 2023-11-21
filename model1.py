from flask import Flask, render_template, Response
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import random

# 모델 로드
model = load_model("keras_Model.h5", compile=False)

# UTF-8 인코딩으로 레이블 로드
class_names = open("labels.txt", "r", encoding="utf-8").readlines()

# keras 모델에 피드할 적절한 모양의 배열 생성
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 카메라에 연결 (0은 일반적으로 기본 카메라)
cap = cv2.VideoCapture(0)

# Flask 어플리케이션 생성
app = Flask(__name__)

# 홈페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 비디오 피드 생성
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # 이미지 처리
            image = Image.fromarray(frame)
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

# 비디오 피드 라우트
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 행동 감지를 위한 변수
action_interval = 20  # 행동을 감지할 간격 (초)
start_time = time.time()
hints_given = 0
rounds_played = 0

# 가위바위보 게임 변수
user_choice = None
computer_choice = None
result = None

# 가위바위보 게임 관련 함수들
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

while True:
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
        print(f"Action Detected! Time: {elapsed_time:.2f}")

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
        action_detected = True
        start_time = time.time()
        hints_given = 0
        rounds_played += 1

        if rounds_played == 5:
            print("5판이 끝났습니다. 이제 인물을 맞춰보세요!")
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("기회가 끝났습니다. 이 사람은 누굴까요?")
user_guess = input("인물의 이름을 입력하세요: ")

if user_guess == "김연아":
    print("정답입니다! 축하합니다!")
else:
    print("아쉽군요. 다시 도전해 보세요!")

cap.release()
cv2.destroyAllWindows()
