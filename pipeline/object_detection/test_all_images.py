from ultralytics import YOLO
from pathlib import Path

# Load model
model = YOLO('weights/best.pt')

# Get all test images
fire_images = list(Path('../test_images/fire').glob('*.jpg'))
nofire_images = list(Path('../test_images/nofire').glob('*.jpg'))

print(f"Testing {len(fire_images)} fire images and {len(nofire_images)} non-fire images\n")

# Test fire images
print("=== FIRE IMAGES ===")
fire_detected = 0
for img in fire_images:
    results = model(str(img), verbose=False)
    num_det = len(results[0].boxes)
    fire_detected += (num_det > 0)
    
    status = "✓ DETECTED" if num_det > 0 else "✗ MISSED"
    if num_det > 0:
        max_conf = max([box.conf[0].item() for box in results[0].boxes])
        print(f"{img.name}: {status} ({num_det} detections, max conf: {max_conf:.2f})")
    else:
        print(f"{img.name}: {status}")

print(f"\nFire detection rate: {fire_detected}/{len(fire_images)} ({100*fire_detected/len(fire_images):.1f}%)")

# Test non-fire images
print("\n=== NON-FIRE IMAGES (Should NOT detect) ===")
false_positives = 0
for img in nofire_images:
    results = model(str(img), verbose=False)
    num_det = len(results[0].boxes)
    false_positives += (num_det > 0)
    
    status = "✗ FALSE POSITIVE" if num_det > 0 else "✓ CORRECT (no detection)"
    if num_det > 0:
        max_conf = max([box.conf[0].item() for box in results[0].boxes])
        print(f"{img.name}: {status} ({num_det} detections, max conf: {max_conf:.2f})")
    else:
        print(f"{img.name}: {status}")

print(f"\nFalse positive rate: {false_positives}/{len(nofire_images)} ({100*false_positives/len(nofire_images):.1f}%)")

print("\n=== SUMMARY ===")
print(f"Fire images correctly detected: {fire_detected}/{len(fire_images)}")
print(f"Non-fire images with false positives: {false_positives}/{len(nofire_images)}")