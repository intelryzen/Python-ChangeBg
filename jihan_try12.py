import cv2
import numpy as np

#동영상 파일 경로 설정
video_path = '디지영 프로젝트 영상/iv17.mov'

#비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

#배경 차분기 초기화
fgbg = cv2.createBackgroundSubtractorMOG2()

def filter_skin_and_black(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 50, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 50], dtype=np.uint8)

    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_or = cv2.bitwise_or(mask_skin, mask_black)

    result = cv2.bitwise_and(image, image, mask=mask_or)

    return result


#동영상 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환

    # 배경 차분을 통한 포그라운드 마스크 생성
    fgmask = filter_skin_and_black(frame)

    cv2.imshow('Foreground Mask', fgmask)

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

#캡처 객체와 윈도우 해제
cap.release()
cv2.destroyAllWindows()