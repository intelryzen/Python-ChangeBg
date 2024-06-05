import cv2
import numpy as np

# 동영상 파일 경로 설정
video_path = "/Users/janhi/DIP/Python-ChangeBg/디지영 프로젝트 영상/iv2.mov"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")

# 배경 차분기 초기화
fgbg = cv2.createBackgroundSubtractorMOG2()

# 출력 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID'는 일반적인 코덱 중 하나입니다.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))


# 동영상 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.4)
    
    # 첫 번째 방법: 엣지 감지
    edges1 = cv2.Canny(gray, 50, 200)

    # 두 번째 방법: 배경 차분을 통한 포그라운드 마스크 생성
    fgmask = fgbg.apply(gray)

    # 마스크 후처리 (노이즈 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # 배경 차분을 통한 엣지 감지
    edges2 = cv2.Canny(fgmask, 1, 10)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사람 영역을 포함하는 윤곽선을 필터링
    person_contours = []
    for contour in contours:
        # 윤곽선의 면적을 계산
        area = cv2.contourArea(contour)
        # 윤곽선의 둘레 길이를 계산
        perimeter = cv2.arcLength(contour, True)
        # 면적이 특정 값 이상이고 둘레 길이가 특정 값 이상인 윤곽선만 남김
        if area > 55:
            person_contours.append(contour)

    # 초기 프레임 복사본
    output_frame = frame.copy()

    # 필터링된 윤곽선 그리기
    contour_mask = np.zeros_like(fgmask)
    cv2.drawContours(contour_mask, person_contours, -1, (255), thickness=cv2.FILLED)

    # D8 거리를 10 이하로 떨어진 점들을 연결하기 위해 팽창 후 축소
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated_contour_mask = cv2.dilate(contour_mask, kernel, iterations=30)


    # 윤곽선 찾기 (팽창 후 축소된 마스크에서)
    final_contours, _ = cv2.findContours(dilated_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 사람 영역을 포함하는 윤곽선을 필터링
    person_contours = []
    for contour in final_contours:
        # 윤곽선의 면적을 계산
        area = cv2.contourArea(contour)
        # 윤곽선의 둘레 길이를 계산
        perimeter = cv2.arcLength(contour, True)
        # 면적이 특정 값 이상이고 둘레 길이가 특정 값 이상인 윤곽선만 남김
        if perimeter > 1800:
            person_contours.append(contour)

    height, width = frame.shape[:2]
    binary_image = np.zeros((height, width), dtype=np.uint8)
    
    # 윤곽선으로 마스크 채우기
    cv2.drawContours(binary_image, person_contours, -1, (255), thickness=cv2.FILLED)

   # 마스크 데이터 타입을 np.uint8로 변환하고 0과 255로 스케일 조정
    binary_image = binary_image.astype(np.uint8) * 255

    # 이미지와 마스크의 크기를 확인
    if binary_image.shape[:2] != frame.shape[:2]:
        # 마스크의 크기를 원본 이미지의 크기에 맞춰 조정
        binary_image = cv2.resize(binary_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

    # 원본 이미지와 마스크를 이용하여 관심 영역만 추출
    masked_image = cv2.bitwise_and(fgmask, fgmask, mask=binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    masked_image = cv2.dilate(masked_image, kernel, iterations=15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    masked_image = cv2.erode(masked_image, kernel, iterations=15)
    ret, binary_image = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY)
    # # 현재 프레임을 그레이스케일로 변환
    # gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # # 엣지 감지
    # edges = cv2.Canny(gray, 100, 250)


    # # 마스크가 적용된 이미지에 Otsu 이진화 적용
    # gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    # ret, otsu_result = cv2.threshold(gray_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # 결과 이미지 출력
    # cv2.imshow('Otsu Thresholding', otsu_result)
    masked_image = cv2.bitwise_and(frame, frame, mask=binary_image)
    out.write(masked_image)


    # 결과 출력
    # cv2.imshow('Final Mask', final_mask)
    cv2.imshow('Person', masked_image)

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap.release()
out.release()
cv2.destroyAllWindows()
