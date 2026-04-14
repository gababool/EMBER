from ultralytics import YOLO

# Load the trained model
model = YOLO('weights/best.pt')

# Test on one fire image
image_path = 'test_image.jpg'

print(f"Testing YOLO on: {image_path}")

# Run detection
results = model(image_path)

# Print results
print(f"\nNumber of detections: {len(results[0].boxes)}")

if len(results[0].boxes) > 0:
    for i, box in enumerate(results[0].boxes):
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        print(f"Detection {i+1}: confidence={conf:.2f}, class={cls}")
else:
    print("No detections found")

# Show the image with bounding boxes
results[0].show()