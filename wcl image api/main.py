import cv2
import numpy as np
from PIL import Image
import torch
import os
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import PyPDF2
import re

# from ultralytics
app = FastAPI()

custom_weights_path = 'last.pt'
conf_thres = 0.25

torch_cache_dir = os.path.join(os.getcwd(), 'torch_cache')
ultralytics_dir = os.path.join(torch_cache_dir, 'Ultralytics')
os.makedirs(torch_cache_dir, exist_ok=True)
os.makedirs(ultralytics_dir, exist_ok=True)

# Set environment variables
os.environ['TORCH_HOME'] = torch_cache_dir
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'MPLCONFIGDIR')
os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.getcwd(), 'YOLO_CONFIG_DIR')

# Create other necessary directories
original_image_folder = os.path.join(os.getcwd(), "original_images")
annotated_image_folder = os.path.join(os.getcwd(), "annotated_images")
os.makedirs(original_image_folder, exist_ok=True)
os.makedirs(annotated_image_folder, exist_ok=True)

# Create folders to save images if they don't exist
original_image_folder = "original_images/"
annotated_image_folder = "annotated_images/"
os.makedirs(original_image_folder, exist_ok=True)
os.makedirs(annotated_image_folder, exist_ok=True)

@app.post("/detect/")
async def detect_objects(image: UploadFile = File(...)):
    # Load the custom YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=custom_weights_path)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print("device used", device)
    # Read image as bytes
    contents = await image.read()

    # Save the original image
    original_image_path = os.path.join(original_image_folder, image.filename)
    with open(original_image_path, "wb") as f:
        f.write(contents)

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

    # List to store heights, widths, and confidence scores for walls
    wall_data = []
    logo_detected = False
    logo_confidence = None
    rkgroup_detected = False
    rkgroup_confidence = None

    # Loop through the detected objects
    for bbox in bboxes:
        x1, y1, x2, y2, conf, class_id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Calculate the width and height of the bounding box
        width = (x2 - x1)
        height = (y2 - y1)

        # Get the label of the detected object
        label = labels[int(class_id)]

        # Check if the detected object is a wall
        if label == "wall":
            # Draw bounding box for walls
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Display the width, height, and confidence score of the bounding box
            cv2.putText(img_np, f'Width: {width}px, Height: {height}px', (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(img_np, f'Label: {label}, Conf: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Append width, height, and confidence score to the list for walls
            feet = "feet"
            wall_data.append({
                "width": f"{width / 30.48:.2f}",
                "height": f"{height / 30.48:.2f}",
                "unit": feet
            })

        # Check if the detected object is a logo
        if label == "logo":
            logo_detected = True
            logo_confidence = conf

        # Check if the detected object is an rkgroup
        if label == "rkgroup":
            rkgroup_detected = True
            rkgroup_confidence = conf

    # Save the annotated image with detections
    annotated_image_path = os.path.join(annotated_image_folder, f"annotated_{image.filename}")
    cv2.imwrite(annotated_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Prepare the JSON response
    response = {"wall_data": wall_data}
    if rkgroup_detected:
        if logo_detected:
            response["Wcl_logo"] = {"status": "logo detected", "confidence": f"{logo_confidence:.2f}"}
        else:
            response["Wcl_logo"] = {"status": "no logo detected"}

        if rkgroup_detected:
            response["rkgroup"] = {"status": "rkgroup detected", "confidence": f"{rkgroup_confidence:.2f}"}
        else:
            response["rkgroup"] = {"status": "no rkgroup detected"}
    else:
        response["wall_data"] = "Not wcl wall"

    # Return JSON response with wall dimensions and logo/rkgroup detection status
    if not wall_data:
        response["wall_data"] = "No wall detected"

    return JSONResponse(content=response)


def extract_details_from_pdf(file: UploadFile):
    pdf_reader = PyPDF2.PdfReader(file.file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    # Extract all document numbers
    doc_number_matches = re.findall(r'Document No\. : (\S+)', text)
    document_numbers = doc_number_matches if doc_number_matches else []

    # Extract all document dates
    doc_date_matches = re.findall(r'Document Date : (\d{2}-\d{2}-\d{4})', text)
    document_dates = doc_date_matches if doc_date_matches else []

    print(f"Extracted Document Numbers: {document_numbers}")  # Print the extracted document numbers
    print(f"Extracted Document Dates: {document_dates}")  # Print the extracted document dates

    return document_numbers, document_dates


@app.post("/IRN_validation")
async def upload_pdf(invoice_number: str = Form(...), invoice_date: str = Form(...), pdf_file: UploadFile = File(...)):
    extracted_invoice_numbers, extracted_invoice_dates = extract_details_from_pdf(pdf_file)

    if not extracted_invoice_numbers:
        raise HTTPException(status_code=400, detail="No invoice numbers found in the PDF.")

    if not extracted_invoice_dates:
        raise HTTPException(status_code=400, detail="No invoice dates found in the PDF.")

    response = {
        "invoice_number": "valid" if invoice_number in extracted_invoice_numbers else "invalid",
        "document_date": "valid" if invoice_date in extracted_invoice_dates else "invalid"
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1012)

