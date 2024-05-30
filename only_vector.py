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
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # 엣지 감지
    edges = cv2.Canny(fgmask, 1, 10)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사람 영역을 포함하는 윤곽선을 필터링
    person_contours = []
    for contour in contours:
        # 윤곽선의 면적을 계산
        area = cv2.contourArea(contour)
        # 윤곽선의 둘레 길이를 계산
        perimeter = cv2.arcLength(contour, True)
        # 면적이 특정 값 이상이고 둘레 길이가 특정 값 이상인 윤곽선만 남김
        person_contours.append(contour)

    # 초기 프레임 복사본
    output_frame = frame.copy()

    # 필터링된 윤곽선 그리기
    contour_mask = np.zeros_like(fgmask)
    cv2.drawContours(contour_mask, person_contours, -1, (255), thickness=cv2.FILLED)

    # 작은 간격을 메우기 위해 윤곽선 팽창 후 축소
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    dilated_contour_mask = cv2.dilate(contour_mask, kernel, iterations=8)
    closed_contour_mask = cv2.erode(dilated_contour_mask, kernel, iterations=1)

    # 윤곽선 찾기 (팽창 후 축소된 마스크에서)
    final_contours, _ = cv2.findContours(closed_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 최종 윤곽선 그리기
    cv2.drawContours(output_frame, final_contours, -1, (0, 255, 0), 2)
    # 윤곽선으로 마스크 채우기
    cv2.drawContours(contour_mask, contours, -1, (255), thickness=cv2.FILLED)

    
    # 결과 출력
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', fgmask)
    # cv2.imshow('Edges', edges)
    cv2.imshow('Contour Mask', contour_mask)
    # cv2.imshow('Closed Contour Mask', closed_contour_mask)  
    cv2.imshow('Filtered Contours', output_frame)

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
