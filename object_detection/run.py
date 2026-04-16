#To run: go into this object_detection folder in terminal and run:
#uv run python3 test_yolo_simple.py
import argparse
import cv2
from ultralytics import YOLO
from model_handler import get_model, ModelLoadError



CLASS_NAMES = {
    0: "Smoke",
    1: "Fire"
}


def run (image_path: str, model_name: str):

        print(f"\nRunning YOLO on: {image_path} (model={model_name})")

        try:
             model = get_model(model_name)
        except ModelLoadError as e:
             print(f"[MODEL ERROR] {e}")
             return
        
        try:
             results = model(image_path)
             result = results[0]
        except Exception as e:
             print(f"[INFERENCE ERROR] failed to process image: {e}")
             return
        


        #override names
        result.names = CLASS_NAMES
        
        #bounding boxes
        boxes = result.boxes

        print(f"\nDetection boxes: {len(boxes)}")

        if len(boxes) == 0:
             print("No detection boxes found")
        else : 
             for i, box in enumerate(boxes):
                  conf = float(box.conf[0])
                  cls = int(box.cls[0])
                  label = CLASS_NAMES.get(cls, str(cls))

                  print(f"{i+1}: {label} | confidence={conf:.2f}")
        result.show()


if __name__ == "__main__": 
     parser = argparse.ArgumentParser()
     parser.add_argument("image_path")
     parser.add_argument("--model", default="test")

     args = parser.parse_args()

     run(args.image_path, args.model)



# Load the trained model
#model = YOLO('weights/best.pt')

#model_best = the best model
#model_last = another model


# 

"""
# Test on one fire image
image_path = 'test_image.jpg'

print(f"Testing YOLO on: {image_path}")
# Run detection
results = model_best(image_path)

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

"""