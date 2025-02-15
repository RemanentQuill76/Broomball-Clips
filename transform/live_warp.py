import cv2
import numpy as np
import json
import sys

if len(sys.argv) != 2:
    print("Usage: python3 live_warp.py <rink_name>")
    sys.exit(1)

rink_name = sys.argv[1]
json_file = f"{rink_name}_warp_config.json"  # Determines which JSON file to load

try:
    with open(json_file, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file {json_file} not found.")
    sys.exit(1)

src_pts = np.float32(config["src_pts"])
dst_pts = np.float32(config["dst_pts"])

# Compute transformation matrix once
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Open webcam (or replace 0 with a video file path for testing)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the precomputed transformation
    warped_frame = cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))

    # Show original and transformed frames side by side
    combined = np.hstack((frame, warped_frame))
    cv2.imshow(f"{rink_name} Rink - Original (Left) | Warped (Right)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

