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

# 새로운 인물 정보 추가
characters = [
    {"name": "강지윤", "hints": ["동글테 안경을 쓴 남학생.", "선생님의 특징을 잘 따라함.", "역시 정상은 없는 루투투.", "6666", "포비"]},
    {"name": "고창환", "hints": ["암기 일짱.", "머리를 짧게 자른 전적이 있음.", "본인 피셜로는 제곽 최고 미남", "'눈만 감고 있었어요'", "원초아임."]},
    {"name": "김나연", "hints": ["김빛나의 애인 중 한 명", "목소리가 좋고 발표를 잘 함.", "염색", "글로브 대빵", "방송부 아나운서"]},
    {"name": "김도율", "hints": ["곱슬 머리에 안경을 끼지 않음.", "초록색 후드집업.", "조졸조진 예상", "외국어 1등..!", "생명+정보"]},
    {"name": "김빛나", "hints": ["안경 쓴 여학생.", "다수의 여친 보유.", "겨울마다 앞머리를 자름.", "게임을 좋아함", "카오스"]},
    {"name": "김아영", "hints": ["뿔테안경을 쓰는 여학생.", "참새 닮음", "케미 일짱", "츤츤데레", "그림을 잘 그림"]},
    {"name": "김채현", "hints": ["키와 덩치가 큼", "대식쌤과 친함", "제곽 헤어스타일 선구자", "시드 양대산맥", "농구, 킨볼 등 운동광"]},
    {"name": "김태이", "hints": ["유산소 선호", "레이업 일짱", "보컬", "뿔테안경 쓴 남학생", "카오스 부장"]},
    {"name": "김현상", "hints": ["친화력 갑", "패셔니스타", "25기 축구 일짱", "브롤 좋아함", "학부모 회장"]},
    {"name": "김효빈", "hints": ["성량이 크고 자신감이 넘침", "안경을 끼지 않는 여학생", "키가 작은 귀요미", "글로브", "선생님 전부를 좋아함"]},
    {"name": "송민준", "hints": ["카메라", "드론", "동수쌤 애제자", "cute aggression", "버츄얼"]},
    {"name": "양승우", "hints": ["25기 최고 인기남", "질투", "축제", "루투투", "그림 잘 그림"]},
    {"name": "유서준", "hints": ["슛 장인", "25기 발로란트 1짱", "'즐기는 게 진정한 바른 학교생활'", "의사"]},
    {"name": "이건우", "hints": ["이세돌", "25기 롤 1짱", "안경 쓴 곱슬머리 남학생", "여친 13명 바람둥이", "케미"]},
    {"name": "이권", "hints": ["'이건 ^말^도 안돼'", "농구 잘함", "안경 쓴 남학생", "곽이부", "체스 1짱"]},
    {"name": "이세호", "hints": ["커먼언커먼", "롤, 옵치, 발로", "농구 잘 함"]},
    {"name": "이예찬", "hints": ["'여친은 못 사귀는 게 아니라 안 사귀는 것.'", "'어디 여자가'", "세탁기 ㅠㅠ", "3D 프린터기", "웹슈터"]},
    {"name": "이은상", "hints": ["헬창", "메이드복 잘 어울릴 것 같은 사람 1위", "개미기두", "체공시간 ㄷㄷ", "유산소는 별로 안 좋아하는 듯..?"]},
    {"name": "이준이", "hints": ["붙임 머리", "냐밍~", "ppt 잘 만드는 사람 1위", "디자인부", "아침 체조"]},
    {"name": "정우진", "hints": ["수업 째기 가능한 사람 1위", "게임충", "깨어 있는 수업시간은 언어뿐", "흑우", "인어공주"]}
]


# 랜덤 인물 선택 함수 추가
def get_random_character():
    return random.choice(characters)

# 힌트 제공 함수 수정
def provide_hint(character):
    global hints_given
    hints = character["hints"]
    if hints_given < len(hints):
        hint = "힌트: " + hints[hints_given]
        hints_given += 1
        return hint
    return ""

# 가위바위보 컴퓨터 선택 함수 추가
def get_computer_choice():
    choices = ["Rock", "Paper", "Scissors"]
    return random.choice(choices)

# 프레임 생성 함수 수정
def gen_frames():
    global cap, model, data, class_names, action_interval, start_time, hints_given, rounds_played
    global user_choice, computer_choice, result, character

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
                result = "이겼습니다! " + provide_hint(character)  # 이기면 힌트 제공
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
                # 다음 게임을 위해 랜덤 인물 선택
                character = get_random_character()
                print(f"인물: {character['name']}")
                break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 인물 초기화 함수 추가
def initialize_character():
    global rounds_played, character
    rounds_played = 0
    # 초기 인물 선택
    character = get_random_character()
    print(f"인물: {character['name']}")

@app.route('/')
def index():
    global result
    return render_template('index.html', result=result)

@app.route('/video_feed')
def video_feed():
    initialize_character()  # 웹 페이지에 처음 접속할 때 초기화
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hint')
def hint():
    global user_choice, computer_choice, result, character
    if result and "이겼습니다!" in result:
        return provide_hint(character)
    return ""

@app.route('/guess', methods=['POST'])
def guess():
    global user_choice, computer_choice, result, character
    user_guess = request.form.get('user_guess')

    # 사용자의 추측이 정답인 경우
    if user_guess == character["name"]:
        result = "정답입니다! 축하합니다!"
    else:
        result = "아쉽군요. 다시 도전해 보세요!"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
