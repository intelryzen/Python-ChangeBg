import cv2
import numpy as np

# 동영상 파일 경로 설정
video_path = '디지영 프로젝트 영상/iv17.mov'
img_path = '디지영 프로젝트 영상/bg.jpg'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)
background_img = cv2.imread(img_path)

# 배경 차분기 초기화
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

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

# 동영상 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (0, 0), 1)

    fgmask = fgbg.apply(gray)
    _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 엣지 감지
    edges = cv2.Canny(gray, 50, 250)

    # 배경 차분을 통한 포그라운드 마스크 생성
    fgmask = fgbg.apply(gray)

    # 마스크 후처리 (노이즈 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=6)

    # 배경 차분을 통한 엣지 감지
    edges_bg = cv2.Canny(fgmask, 50, 250)

    # 팽창 후 축소로 윤곽선 연결 및 채우기
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_edges = cv2.dilate(edges_bg, kernel, iterations=15)
    closed_edges = cv2.erode(dilated_edges, kernel, iterations=1)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사람 영역을 포함하는 윤곽선을 필터링
    person_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]

    # 초기 프레임 복사본
    output_frame = frame.copy()

    # 필터링된 윤곽선 그리기
    contour_mask = np.zeros_like(fgmask)
    cv2.drawContours(contour_mask, person_contours, -1, (255), thickness=cv2.FILLED)

    # 팽창 후 축소로 윤곽선 연결 및 채우기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_contour_mask = cv2.dilate(contour_mask, kernel, iterations=10)
    closed_contour_mask = cv2.erode(dilated_contour_mask, kernel, iterations=5)

    # 최종 윤곽선 찾기
    final_contours, _ = cv2.findContours(closed_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 최종 윤곽선으로 마스크 채우기
    final_mask = np.zeros_like(fgmask)
    cv2.drawContours(final_mask, final_contours, -1, (255), thickness=cv2.FILLED)

    # 마스크 적용하여 사람만 추출
    person = cv2.bitwise_and(frame, frame, mask=final_mask)

    # 배경 이미지 크기 조정 (동영상 프레임 크기와 일치)
    background_resized = cv2.resize(background_img, (frame.shape[1], frame.shape[0]))

    # 사람 부분을 배경 이미지에 복사
    inverse_mask = cv2.bitwise_not(final_mask)
    background_part = cv2.bitwise_and(background_resized, background_resized, mask=inverse_mask)
    combined_frame = cv2.add(background_part, person)

    # result_final = filter_skin_and_black(combined_frame)

    # cv2.imshow('Foreground Mask', result_final)

    

    # 결과 출력
    cv2.imshow('Person in New Background', combined_frame)

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
