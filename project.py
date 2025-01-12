import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20)

# Kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

# Minimum area for display
MIN_AREA = 20000
PADDING = 20  # Padding for each grouped box

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply background subtraction
    fgmask = fgbg.apply(blurred_frame)

    # Apply morphological operations to reduce noise and close gaps
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the detected motion
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect bounding boxes for detected contours
    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 10000:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append([x, y, w, h])

    # If there are bounding boxes, apply DBSCAN clustering
    if bounding_boxes:
        # Convert bounding boxes to a format compatible with DBSCAN (center points)
        centers = np.array([[x + w / 2, y + h / 2] for x, y, w, h in bounding_boxes])

        # Apply DBSCAN clustering with a higher `eps` for grouping distance
        clustering = DBSCAN(eps=700, min_samples=1).fit(centers)

        # Group bounding boxes by cluster labels
        grouped_boxes = {}
        for i, label in enumerate(clustering.labels_):
            if label not in grouped_boxes:
                grouped_boxes[label] = []
            grouped_boxes[label].append(bounding_boxes[i])

        # Draw grouped bounding boxes, filtering out small boxes
        for box_group in grouped_boxes.values():
            # Calculate the combined bounding box for each cluster
            x_min = min([box[0] for box in box_group]) - PADDING
            y_min = min([box[1] for box in box_group]) - PADDING
            x_max = max([box[0] + box[2] for box in box_group]) + PADDING
            y_max = max([box[1] + box[3] for box in box_group]) + PADDING

            # Calculate the area of the combined bounding box
            area = (x_max - x_min) * (y_max - y_min)

            # Only draw the bounding box if it's above the minimum area
            if area > MIN_AREA:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, f"X: {x_min}, Y: {y_min}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Feed - Object Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()