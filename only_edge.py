import cv2
import numpy as np

# 동영상 파일 경로 설정
video_path = '디지영 프로젝트 영상/iv1.mov'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 첫 번째 프레임 읽기
ret, frame = cap.read()


def connect_close_points(image, max_distance=3):
    # 이미지의 높이와 너비 추출
    height, width = image.shape
    
    # 출력 이미지 준비 (컬러로 변경)
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 모든 픽셀 위치를 저장하는 배열 준비
    points = [(x, y) for x in range(width) for y in range(height) if image[y, x] == 255]
    
    # 모든 점 쌍에 대해 거리 계산
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            point1 = points[i]
            point2 = points[j]
            distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
            if distance <= max_distance:
                cv2.line(output_image, point1, point2, (255, 0, 0), 1)  # 선으로 연결
    
    return output_image

# 동영상 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 엣지 감지
    edges = cv2.Canny(gray, 100, 250)


    # 점 연결 및 이미지 출력
    connected_image = connect_close_points(edges)
    # 결과 출력
    # cv2.imshow('Original', frame)
    cv2.imshow('Edges', connected_image)

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
