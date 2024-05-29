import cv2
import numpy as np

# Initialize the background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Open the input video
input_video = cv2.VideoCapture('test.mp4')

# Load the background image
background_img = cv2.imread('goldhill.bmp')
if background_img is None:
    raise ValueError("Background image not found or cannot be read")

# Get the video frame dimensions and FPS
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)

# Ensure FPS is a positive integer
if fps <= 0:
    fps = 30  # Default FPS if the input video FPS is not valid

# Resize the background image to match the frame size
background_img = cv2.resize(background_img, (frame_width, frame_height))

# Define the codec and create VideoWriter object to save the output video
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define the kernel for morphological operations
kernel = np.ones((25, 25), np.uint8)

while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break
    
    # Apply the background subtractor to get the foreground mask
    fg_mask = back_sub.apply(frame)
    
    # Threshold the mask to remove shadows (if any)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise and fill gaps
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)   
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)   
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)   
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)   
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)   

    # Optionally, apply Gaussian blur to smooth the edges
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an output frame by copying the background image
    output_frame = background_img.copy()
    
    # Draw the detected moving person(s) on the output frame
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            # Create a mask for the current contour
            mask = np.zeros_like(fg_mask)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            mask = mask.astype(bool)
            
            # Use the mask to copy the detected person to the output frame
            output_frame[mask] = frame[mask]
    
    # Write the output frame to the video
    output_video.write(output_frame)
    
    # Optionally, display the frames
    cv2.imshow('Input', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Output', output_frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video objects and close all windows
input_video.release()
output_video.release()
cv2.destroyAllWindows()
