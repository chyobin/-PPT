document.addEventListener('DOMContentLoaded', function() {
    const gameResultsDiv = document.getElementById('game-results');
    
    // 게임 결과를 업데이트하는 함수
    function updateGameResults(data) {
        const userChoiceParagraph = document.getElementById('user-choice');
        const computerChoiceParagraph = document.getElementById('computer-choice');
        const resultParagraph = document.getElementById('result');

        userChoiceParagraph.textContent = 'User Choice: ' + data.user_choice;
        computerChoiceParagraph.textContent = 'Computer Choice: ' + data.computer_choice;
        resultParagraph.textContent = 'Result: ' + data.result;
    }

    // 1초마다 `/play_game` 엔드포인트를 호출하여 게임 결과 업데이트
    setInterval(function() {
        fetch('/play_game')
            .then(response => response.json())
            .then(data => {
                if (!data.error) {
                    updateGameResults(data);
                }
            })
            .catch(error => console.error('Error:', error));
    }, 1000);
});
