import cv2
import numpy as np

video_path = "/Users/janhi/DIP/Python-ChangeBg/디지영 프로젝트 영상/iv17.mov"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height), True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)

    fgmask = fgbg.apply(gray)
    _, fgmask = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 250)
    fgmask = cv2.bitwise_and(fgmask, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=15)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=3)
    


    num_labels, labels_im = cv2.connectedComponents(fgmask)
    if num_labels > 1:
        component_sizes = np.bincount(labels_im.flatten())[1:]
        for component in range(1, num_labels):
            if component_sizes[component - 1] > 100:
                component_mask = np.uint8(labels_im == component) * 255
                person = cv2.bitwise_and(frame, frame, mask=component_mask)
                out.write(person)
                cv2.imshow('Jumping Person', person)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
