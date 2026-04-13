# enhanced_detect_objects.py - Improved version with more features!

from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("yolo11n.pt")

# Run detection and save results
results = model.predict(
    source="test_image.jpg",
    save=True,  # Save the result image
    save_txt=True,  # Save detection details
    conf=0.0001  # Only show detections with 50%+ confidence
)

# Print detailed information about detections
for r in results:
    print(f"\n🎯 Found {len(r.boxes)} objects in your image!")
    
    for i, box in enumerate(r.boxes):
        # Get object details
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        
        print(f"  {i+1}. {class_name} ({confidence:.1%} confidence)")
    
    # Show the image
    r.show()

print("\n✅ Results saved in 'runs/detect/predict/' folder!")