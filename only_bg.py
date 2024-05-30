import cv2
import numpy as np

# 동영상 파일 경로 설정
video_path = 'iv1.mov'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 배경 차분기 초기화
fgbg = cv2.createBackgroundSubtractorMOG2()

# 동영상 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 배경 차분을 통한 포그라운드 마스크 생성
    fgmask = fgbg.apply(gray)

    # 마스크 후처리 (노이즈 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # 결과 출력
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fgmask)

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
