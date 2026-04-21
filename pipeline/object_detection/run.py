#To run: go into this object_detection folder in terminal and run:
#uv run python run.py test_image.jpg --model best
#best is the default model, you can also replace that with test, model_3 etc.

#To run from the repo root:
# uv run python pipeline/object_detection/run.py <image_path> --model <model_name>

# no window of result (dont open image)
#uv run python pipeline/object_detection/run.py <image_path> --model <model_name>

# with window of result (open image)
# uv run python pipeline/object_detection/run.py <image_path> --model <model_name> --show

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import cv2
from ultralytics import YOLO
from model_handler import get_model, ModelLoadError
from config import ANNOTATED_OUTPUT_DIR, ANNOTATED_OUTPUT_FILENAME



sys.path.insert(0, str(Path(__file__).resolve().parent))

CLASS_NAMES = {
    0: "Smoke",
    1: "Fire"
}


def run (image_path: str, model_name: str, show: bool):

        print(f"\nRunning YOLO on: {image_path} (model={model_name})")

        #image path
        image_path = Path(image_path).resolve()

        if not image_path.exists():
             print(f"[IMAGE_ERROR] Image not found: {image_path}")
             return

        try:
             model = get_model(model_name)
        except ModelLoadError as e:
             print(f"[MODEL ERROR] {e}")
             return
        
        try:
             results = model(str(image_path))
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
        if show:
             result.show()
          # save annotated image
        ANNOTATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = ANNOTATED_OUTPUT_DIR / ANNOTATED_OUTPUT_FILENAME
        annotated = result.plot()
        cv2.imwrite(str(output_path), annotated)
        print(f"\nAnnotated image saved to: {output_path}")  





if __name__ == "__main__": 
     parser = argparse.ArgumentParser()
     parser.add_argument("image_path")
     parser.add_argument("--model", default="best")
     parser.add_argument("--show", action="store_true" )

     args = parser.parse_args()

     run(args.image_path, args.model, args.show)



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