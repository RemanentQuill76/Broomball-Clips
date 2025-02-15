import cv2
import numpy as np
import json
import sys

if len(sys.argv) != 4:
    print("Usage: python3 mp4_warp.py <rink_name> <input_video> <output_video>")
    sys.exit(1)

rink_name = sys.argv[1]
input_video = sys.argv[2]
output_video = sys.argv[3]

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

# Open video file
cap = cv2.VideoCapture(input_video)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the precomputed transformation
    warped_frame = cv2.warpPerspective(frame, matrix, (frame_width, frame_height))

    # Write transformed frame
    out.write(warped_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Transformed video saved as {output_video}")

