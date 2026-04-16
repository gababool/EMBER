#To run: go into this object_detection folder in terminal and run:
#uv run python3 test_yolo_simple.py

import cv2
from ultralytics import YOLO
from config import model_next_best
from config import model_last
from config import model_best
from config import model_test


# Load the trained model
#model = YOLO('weights/best.pt')

#model_best = the best model
#model_last = another model



# Test on one fire image
image_path = '43772086021_c501debc58_o.jpg'

print(f"Testing YOLO on: {image_path}")
# Run detection
results = model_test(image_path)

# Override the names on the result before displaying
results[0].names = {
    0: 'Smoke',
    1: 'Fire'
}

# Print results
print(f"\nNumber of detections: {len(results[0].boxes)}")
print(model_best.names)
if len(results[0].boxes) > 0:
    for i, box in enumerate(results[0].boxes):
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = results[0].names[cls]
        print(f"Detection {i+1}: confidence={conf:.2f}, class={cls} ({label})")
else:
    print("No detections found")

# Show the image with bounding boxes
results[0].show()

