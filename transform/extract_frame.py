import cv2
import numpy as np
import sys
import json

if len(sys.argv) != 3:
    print("Usage: python3 extract_frame.py <input_video> <frame_time_seconds>")
    sys.exit(1)

video_path = sys.argv[1]
frame_time = float(sys.argv[2])  # Time in seconds

# Open video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = int(fps * frame_time)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Move to frame

ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not extract frame.")
    sys.exit(1)

# Padding applied
PAD_X = 400  # Extra width
PAD_Y = 400  # Extra height
canvas = np.zeros((frame.shape[0] + PAD_Y, frame.shape[1] + PAD_X, 3), dtype=np.uint8)
canvas[PAD_Y//2 : PAD_Y//2 + frame.shape[0], PAD_X//2 : PAD_X//2 + frame.shape[1]] = frame

frame_filename = "sample_frame_padded.jpg"
cv2.imwrite(frame_filename, canvas)
print(f"Frame saved as {frame_filename}")

# Store adjusted points
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust clicked points back to the original frame coordinates
        adjusted_x = x - (PAD_X // 2)
        adjusted_y = y - (PAD_Y // 2)

        # Ensure values remain within original frame size
        adjusted_x = max(0, min(adjusted_x, frame.shape[1] - 1))
        adjusted_y = max(0, min(adjusted_y, frame.shape[0] - 1))

        points.append((adjusted_x, adjusted_y))
        print(f"Original Frame Point Selected: ({adjusted_x}, {adjusted_y})")

        if len(points) == 4:
            cv2.destroyAllWindows()

# Display the padded image
cv2.imshow("Click 4 Points (Includes Extra Space)", canvas)
cv2.setMouseCallback("Click 4 Points (Includes Extra Space)", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nFinal Adjusted Points (Mapped to Original Frame):")
print(points)

# Save points to JSON format
warp_config = {"src_pts": points}
with open("warp_config_extracted.json", "w") as f:
    json.dump(warp_config, f, indent=4)

print("\nSaved adjusted transformation points to warp_config_extracted.json")

