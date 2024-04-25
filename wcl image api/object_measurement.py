import cv2
import numpy as np
from PIL import Image
import torch
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

custom_weights_path = 'last 1.pt'
image_path = "images/"
conf_thres = 0.25

@app.post("/upload")
# Function to load custom YOLOv5 model and perform object detection
async def detect_objects(image: UploadFile = File(...)):
    # Load your custom YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=custom_weights_path)

    # Read image as bytes
    contents = await image.read()

    # Convert bytes to numpy array
    nparr = np.frombuffer(contents, np.uint8)

    # Decode numpy array into image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert image to PIL format
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Perform object detection
    results = model(img, size=640)  # Adjust size if necessary

    # Extract bounding boxes, class predictions, and corresponding labels
    bboxes = results.xyxy[0].cpu().numpy()  # Convert bounding boxes to NumPy array
    labels = results.names[0]  # Extract class labels from the model

    # Convert PIL image to OpenCV format (numpy array)
    img_np = np.array(img)

    # Loop through the detected objects
    for bbox in bboxes:
        x1, y1, x2, y2, conf, class_id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate the width and height of the bounding box
        width = x2 - x1
        height = y2 - y1

        # Draw bounding box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the width and height of the bounding box
        cv2.putText(img_np, f'Width: {width}, Height: {height}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Add wall label
        label = labels[int(class_id)]
        cv2.putText(img_np, label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    annotated_folder = "annotated_images/"
    # Save the annotated image with detections
    annotated_image_path = annotated_folder+  "annotated_" + image.filename
    cv2.imwrite(annotated_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    # annotated_image_path = detect_objects(annotated_image_path, custom_weights_path)
    # img = cv2.imread(annotated_image_path)
    # return annotated_image_path
    
    return JSONResponse(content={"width": str(width)+" cm", "height": str(height)+" cm"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
