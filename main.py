# import cv2
# from ultralytics import YOLO

# # Load the model
# model = YOLO("yolov8n.pt")

# # Start capturing video from the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
    
#     # Inference
#     results = model(frame)

#     # Plot results on the frame 
#     annotated_frame = results[0].plot()
    
#     # Display the frame with bounding boxes and labels
#     cv2.imshow("YOLOv8 Object Detection", annotated_frame)
    
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all OpenCV windows 
# cap.release()
# cv2.destroyAllWindows()


import cv2
import time
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# For FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Record the start time
    current_time = time.time()

    # Inference
    results = model(frame)

    # Plot the results
    annotated_frame = results[0].plot()

    # Calculate FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Put FPS text on the frame
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
