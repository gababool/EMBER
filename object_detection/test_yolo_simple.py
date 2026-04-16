#To run: go into this object_detection folder in terminal and run:
#uv run python3 test_yolo_simple.py

from ultralytics import YOLO
from config import model_best
from config import model_last

# Load the trained model
#model = YOLO('weights/best.pt')

#model_best = the best model
#model_last = another model

# Test on one fire image
image_path = 'test_image.jpg'

print(f"Testing YOLO on: {image_path}")

# Run detection
results = model_best(image_path)

# Print results
print(f"\nNumber of detections: {len(results[0].boxes)}")


if len(results[0].boxes) > 0:
    for i, box in enumerate(results[0].boxes):
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        print(f"Detection {i+1}: confidence={conf:.2f}, class={cls}")
        #Class (cls) = 1 is smoke and 0 is fire
else:
    print("No detections found")

# Show the image with bounding boxes
results[0].show()