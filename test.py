import cv2
import numpy as np

# 동영상 파일 경로 설정
video_path = 'iv1.mov'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 첫 번째 프레임 읽기
ret, frame = cap.read()

# 동영상 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 엣지 감지
    edges = cv2.Canny(gray, 100, 250)

    # 결과 출력
    cv2.imshow('Original', frame)
    cv2.imshow('Edges', edges)

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
