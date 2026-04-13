

# Import the YOLO model
from ultralytics import YOLO

# Load the pre-trained YOLO model (this downloads automatically)
model = YOLO("yolo11n.pt")

# Predict objects in your image
results = model.predict(source="test_image.jpg")

# Show the results with bounding boxes
for r in results:
    r.show()
