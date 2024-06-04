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
async def detect_objects(image: UploadFile = File(...)):
    # Load the custom YOLOv5 model
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
    labels = results.names  # Extract class labels from the model

    # Convert PIL image to OpenCV format (numpy array)
    img_np = np.array(img)

    # List to store heights, widths, labels, and confidence scores for this image
    image_data = []

    # Loop through the detected objects
    for bbox in bboxes:
        x1, y1, x2, y2, conf, class_id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate the width and height of the bounding box
        width = (x2 - x1)
        height = (y2 - y1)

        # Draw bounding box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the width, height, label, and confidence score of the bounding box
        label = labels[int(class_id)]
        conf_text = f"{conf:.2f}"
        cv2.putText(img_np, f'Width: {width}px, Height: {height}px', (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img_np, f'Label: {label}, Conf: {conf_text}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Append width, height, label, and confidence score to the list for this image
        image_data.append({
            "width": f"{width / 30.48:.2f} feet",
            "height": f"{height / 30.48:.2f} feet",
            "label": label,
            "confidence": conf_text
        })

    # Define folder to save annotated images
    annotated_folder = "annotated_images/"
    if not os.path.exists(annotated_folder):
        os.makedirs(annotated_folder)

    # Save the annotated image with detections
    annotated_image_path = os.path.join(annotated_folder, f"annotated_{image.filename}")
    cv2.imwrite(annotated_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Return JSON response with detected dimensions or error message
    if not image_data:
        return JSONResponse(content={"error": "No wall detected"})
    else:
        return JSONResponse(content={"image_data": image_data})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
