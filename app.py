# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response, request
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import time
import random

app = Flask(__name__)

# 컴퓨터의 선택을 랜덤으로 반환하는 함수
def get_computer_choice():
    return random.choice(["Rock", "Paper", "Scissors"])

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
random_character = None  # 추가된 부분
character_hints = {}  # 추가된 부분

# 힌트 제공 함수
def provide_hint():
    global hints_given, random_character, character_hints
    if hints_given < len(character_hints.get(random_character, [])):
        hint_to_return = "힌트: " + character_hints[random_character][hints_given]
        hints_given += 1
        return hint_to_return
    return ""

# 가위바위보 게임에서 랜덤 인물 선택 함수
def get_random_character():
    characters = [
        "강지윤", "고창환", "김나연", "김도율", "김빛나",
        "김아영", "김채현", "김태이", "김현상", "김효빈",
        "송민준", "양승우", "유서준", "이건우", "이권",
        "이세호", "이예찬", "이은상", "이준이", "정우진"
    ]
    return random.choice(characters)

# 인물 힌트 초기화 함수
def initialize_character_hints():
    global character_hints
    character_hints = {
    "강지윤": [
        "동글테 안경을 쓴 남학생입니다.",
        "선생님의 특징을 잘 따라합니다.",
        "루투투입니다.",
        "6666",
        "포비"
    ],
    "고창환": [
        "암기 일짱입니다.",
        "머리를 짧게 자른 전적이 있습니다.",
        "본인 피셜에 따르면 제곽 최고 미남",
        "'눈만 감고 있었어요'",
        "원초아임"
    ],
    "김나연": [
        "김빛나의 애인 중 한 명입니다.",
        "목소리가 좋고 발표를 잘합니다.",
        "염색을 즐깁니다.",
        "글로브 대빵",
        "방송부 아나운서"
    ],
    "김도율": [
        "곱슬 머리에 안경을 끼지 않는 학생입니다.",
        "초록색 후드집업을 자주 입습니다.",
        "조졸조진 예상",
        "외국어 1등입니다!",
        "생명+정보"
    ],
    "김빛나": [
        "안경 쓴 여학생입니다.",
        "다수의 여친을 보유하고 있습니다.",
        "겨울마다 앞머리를 자릅니다.",
        "게임을 좋아합니다.",
        "카오스"
    ],
    "김아영": [
        "뿔테안경을 쓰는 여학생입니다.",
        "참새와 닮은 얼굴입니다.",
        "케미 일짱",
        "츤츤데레",
        "그림을 잘 그립니다."
    ],
    "김채현": [
        "키와 덩치가 큰 학생입니다.",
        "대식쌤과 친합니다.",
        "제곽 헤어스타일의 선구자입니다.",
        "시드 양대산맥",
        "농구, 킨볼 등 운동을 좋아합니다."
    ],
    "김태이": [
        "유산소를 선호합니다.",
        "레이업 일짱입니다.",
        "보컬 실력이 뛰어납니다.",
        "뿔테안경을 쓴 남학생입니다.",
        "카오스 부장"
    ],
    "김현상": [
        "친화력이 뛰어납니다.",
        "패셔니스타입니다.",
        "25기 축구 일짱입니다.",
        "브롤을 좋아합니다.",
        "학부모 회장"
    ],
    "김효빈": [
        "성량이 크고 자신감이 넘칩니다.",
        "안경을 끼지 않는 여학생입니다.",
        "키가 작은 귀요미입니다.",
        "글로브를 항상 가지고 다닙니다.",
        "선생님 전부를 좋아합니다."
    ],
    "송민준": [
        "카메라와 드론을 좋아합니다.",
        "동수쌤의 애제자입니다.",
        "cute aggression이 강합니다.",
        "가상 세계에 빠져 삽니다.",
        "버츄얼"
    ],
    "양승우": [
        "25기에서 최고의 인기를 끌고 있는 남학생입니다.",
        "질투심이 강합니다.",
        "축제를 즐깁니다.",
        "루투투입니다.",
        "그림 잘 그립니다."
    ],
    "유서준": [
        "슛 장인입니다.",
        "25기 발로란트 1짱입니다.",
        "'즐기는 게 진정한 바른 학교생활'",
        "의사가 꿈입니다."
    ],
    "이건우": [
        "이세돌과 같은 이름을 가진 학생입니다.",
        "25기에서 롤 1짱입니다.",
        "안경 쓴 곱슬머리 남학생입니다.",
        "여친 13명을 거친 바람둥이입니다.",
        "케미"
    ],
    "이권": [
        "'이건 말도 안돼'라는 대사로 유명한 학생입니다.",
        "농구를 잘합니다.",
        "안경 쓴 남학생입니다.",
        "곽이부",
        "체스 1짱"
    ],
    "이세호": [
        "커먼언커먼입니다.",
        "롤, 옵치, 발로를 즐깁니다.",
        "농구를 잘합니다."
    ],
    "이예찬": [
        "'여친은 못 사귀는 게 아니라 안 사귀는 것'이라는 말로 유명한 학생입니다.",
        "'어디 여자가'라는 대사로 유명합니다.",
        "세탁기를 좋아합니다.",
        "3D 프린터기를 사용합니다.",
        "웹슈터"
    ],
    "이은상": [
        "헬창입니다.",
        "메이드복이 잘 어울릴 것 같은 사람 1위입니다.",
        "개미기두를 갖고 있습니다.",
        "체공시간이 높습니다.",
        "유산소를 별로 좋아하지 않습니다."
    ],
    "이준이": [
        "붙임 머리를 한 학생입니다.",
        "냐밍~",
        "PPT를 잘 만드는 사람 1위입니다.",
        "디자인부에 속해 있습니다.",
        "아침에 체조를 합니다."
    ],
    "정우진": [
        "수업을 째는 것이 가능한 학생 1위입니다.",
        "게임에 빠져 삽니다.",
        "수업시간에는 언어만 깨어 있습니다.",
        "흑우",
        "인어공주"
    ]
}


# 초기 힌트를 설정
initialize_character_hints()

def gen_frames():
    global cap, model, data, class_names, action_interval, start_time, hints_given, rounds_played
    global user_choice, computer_choice, result, random_character

    action_detected = False  # action_detected 변수 초기화

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
        if confidence_score > 0.5 and elapsed_time >= action_interval and not action_detected:
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
            if rounds_played >= 5:
                print("5판이 끝났습니다. 이제 인물을 맞춰보세요!")
                random_character = get_random_character()
                print("랜덤 인물:", random_character)
            else:
                # 이기면 힌트를 제공하고 템플릿에서 힌트를 사용할 수 있도록 result에 추가
                if result and "이겼습니다!" in result:
                    hint = provide_hint()
                    result += " " + hint

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ... (이후 코드)

@app.route('/')
def index():
    global result
    return render_template('index.html', result=result)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hint')
def hint():
    global result, hints_given, random_character

    # Check if the user won and provide hints
    if result and "이겼습니다!" in result:
        return provide_hint()

    # Check if the user is guessing the character and provide hints
    elif result and "랜덤 인물" in result:
        return provide_hint()

    return ""

# 힌트 제공 함수
def provide_hint():
    global hints_given, random_character, character_hints
    if random_character and hints_given < len(character_hints.get(random_character, [])):
        hint_to_return = "힌트: " + character_hints[random_character][hints_given]
        hints_given += 1
        return hint_to_return
    return ""

@app.route('/guess', methods=['POST'])
def guess():
    global user_choice, computer_choice, result, random_character
    user_guess = request.form.get('user_guess')
    
    # 사용자의 추측이 정확한지 확인
    if user_guess == random_character:
        result = "정답입니다! 축하합니다!"
    else:
        result = "아쉽군요. 다시 도전해 보세요!"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
