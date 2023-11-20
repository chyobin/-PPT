const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultDiv = document.getElementById('result');
const hintDiv = document.getElementById('hint');
const ctx = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((error) => {
        console.error('웹캠에 액세스하는 중 오류 발생:', error);
    });

// 여기에 TensorFlow.js 및 게임 로직 추가
