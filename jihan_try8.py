import cv2
import numpy as np

video_path = "/Users/janhi/DIP/Python-ChangeBg/디지영 프로젝트 영상/iv17.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=0, detectShadows=False)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height), True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for background subtraction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    fgmask = fgbg.apply(gray)

    # Convert to HSV and threshold to remove shadows
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    shadow_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))  # Adjust the upper HSV values to suit your needs

    # Combine the shadow mask with the foreground mask
    fgmask = cv2.bitwise_and(fgmask, fgmask, mask=shadow_mask)

    # Apply Canny edge detector
    edges = cv2.Canny(gray, 50, 250)
    fgmask = cv2.bitwise_and(fgmask, edges)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=20)
    fgmask = cv2.dilate(fgmask, kernel, iterations=5)

    # Connected Components to find the person
    num_labels, labels_im = cv2.connectedComponents(fgmask)
    if num_labels > 1:
        component_sizes = np.bincount(labels_im.flatten())[1:]
        largest_component = np.argmax(component_sizes) + 1
        person_mask = np.uint8(labels_im == largest_component) * 255

        # Isolate the person using the mask
        person = cv2.bitwise_and(frame, frame, mask=person_mask)
        out.write(person)
        cv2.imshow('Jumping Person', person)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
