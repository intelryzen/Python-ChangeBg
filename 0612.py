import cv2
import numpy as np

video_path = '디지영 프로젝트 영상/iv2.mov'

cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

img_path = 'bg.jpg'
background_img = cv2.imread(img_path)
background_img = cv2.resize(background_img, (frame_width, frame_height))

def filter_skin_and_black(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 50, 90], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 255, 90], dtype=np.uint8)
    
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    mask_or = cv2.bitwise_or(mask_skin, mask_black)
    
    result = cv2.bitwise_and(image, image, mask=mask_or)
    
    return result

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.4)
    
    fgmask = fgbg.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(fgmask, 1, 10)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 55:
            person_contours.append(contour)


    # 필터링된 윤곽선 그리기
    contour_mask = np.zeros_like(fgmask)
    cv2.drawContours(contour_mask, person_contours, -1, (255), thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated_contour_mask = cv2.dilate(contour_mask, kernel, iterations=30)
    final_contours, _ = cv2.findContours(dilated_contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_contours = []
    for contour in final_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 1800:
            person_contours.append(contour)

    height, width = frame.shape[:2]
    binary_image = np.zeros((height, width), dtype=np.uint8)
    
    cv2.drawContours(binary_image, person_contours, -1, (255), thickness=cv2.FILLED)
    binary_image = binary_image.astype(np.uint8) * 255

    if binary_image.shape[:2] != frame.shape[:2]:
        binary_image = cv2.resize(binary_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

    masked_image = cv2.bitwise_and(fgmask, fgmask, mask=binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    masked_image = cv2.dilate(masked_image, kernel, iterations=15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    masked_image = cv2.erode(masked_image, kernel, iterations=15)
    _, binary_image = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY)

    color = cv2.bitwise_and(frame, frame, mask=binary_image)
    color = filter_skin_and_black(color)

    gray = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=7)
    color = cv2.bitwise_and(frame, frame, mask=binary_image)

    ############################################################
    # 배경 이미지와 합성
    inverse_mask = cv2.bitwise_not(binary_image)
    background_part = cv2.bitwise_and(background_img, background_img, mask=inverse_mask)
    final_frame = cv2.add(background_part, color)

    out.write(final_frame)
    cv2.imshow('Person', final_frame)
    ##############################################################

    # q 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 캡처 객체와 윈도우 해제
cap.release()
out.release()
cv2.destroyAllWindows()
