var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', function() {
    console.log('Socket connected');
});

socket.on('disconnect', function() {
    console.log('Socket disconnected');
});

socket.on('update', function(data) {
    // 서버에서의 업데이트 처리
    // ...
});

function getHint() {
    fetch('/hint')
        .then(response => response.text())
        .then(hint => {
            if (hint) {
                alert(hint);
            } else {
                alert("힌트를 얻을 수 없습니다.");
            }
        });
}
