# # import cv2
# # import numpy as np
# # from PIL import Image
# # import torch
# # import os
# #
# #
# # # Function to load custom YOLOv5 model and perform object detection
# # def detect_objects(image_path: str, custom_weights_path: str, conf_thres: float = 0.25):
# #     # Load your custom YOLOv5 model
# #     model = torch.hub.load("ultralytics/yolov5", "custom", path=custom_weights_path)
# #
# #     # Load the image
# #     img = Image.open(image_path)
# #
# #     # Perform object detection
# #     results = model(img, size=640)  # Adjust size if necessary
# #
# #     # Extract bounding boxes and class predictions
# #     bboxes = results.xyxy[0].cpu().numpy()  # Convert bounding boxes to NumPy array
# #
# #     # Loop through the detected objects
# #     for bbox in bboxes:
# #         x1, y1, x2, y2, conf, class_id = bbox
# #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #
# #         # Calculate the width and height of the bounding box
# #         width = x2 - x1
# #         height = y2 - y1
# #
# #         # Draw bounding box
# #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #
# #         # Display the width and height of the bounding box
# #         cv2.putText(img, f'Width: {width}, Height: {height}', (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# #
# #     # Save the annotated image with detections
# #     annotated_image_path = "annotated_" + os.path.basename(image_path)
# #     img.save(annotated_image_path)


# #
# #     return annotated_image_path
# #
# #
# # # Load the image
# # image_path = r'C:\Wall measurement new\WCL\image_0.jpg'
# #
# # # Load your custom YOLOv5 model weights
# # custom_weights_path = r'C:\Wall measurement new\WCL\runs\train\exp10\weights\last.pt'
# #
# # # Perform object detection and measure object size
# # annotated_image_path = detect_objects(image_path, custom_weights_path)
# #
# # # Display the annotated image
# # img = cv2.imread(annotated_image_path)
# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()




# #
# # import cv2
# # import numpy as np
# # from PIL import Image
# # import torch
# # import os
# #
# #
# # # Function to load custom YOLOv5 model and perform object detection
# # def detect_objects(image_path: str, custom_weights_path: str, conf_thres: float = 0.25):
# #     # Load your custom YOLOv5 model
# #     model = torch.hub.load("ultralytics/yolov5", "custom", path=custom_weights_path)
# #
# #     # Load the image
# #     img = Image.open(image_path)
# #
# #     # Perform object detection
# #     results = model(img, size=640)  # Adjust size if necessary
# #
# #     # Extract bounding boxes and class predictions
# #     bboxes = results.xyxy[0].cpu().numpy()  # Convert bounding boxes to NumPy array
# #
# #     # Loop through the detected objects
# #     for bbox in bboxes:
# #         x1, y1, x2, y2, conf, class_id = bbox
# #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #
# #         # Calculate the width and height of the bounding box
# #         width = x2 - x1
# #         height = y2 - y1
# #
# #         # Draw bounding box
# #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #
# #         # Display the width and height of the bounding box
# #         cv2.putText(img, f'Width: {width}, Height: {height}', (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# #
# #     # Convert image to RGB mode
# #     img = img.convert("RGB")
# #
# #     # Save the annotated image with detections
# #     annotated_image_path = "annotated_" + os.path.basename(image_path)
# #     img.save(annotated_image_path)
# #
# #     return annotated_image_path
# #
# #
# # # Load the image
# # image_path = r'C:\Wall measurement new\WCL\20240321_161547.jpg'
# #
# # # Load your custom YOLOv5 model weights
# # custom_weights_path = r'C:\Wall measurement new\WCL\runs\train\exp10\weights\last.pt'
# #
# # # Perform object detection and measure object size
# # annotated_image_path = detect_objects(image_path, custom_weights_path)
# #
# # # Display the annotated image
# # img = cv2.imread(annotated_image_path)
# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # import cv2
# # import numpy as np
# # from PIL import Image
# # import torch
# # import os
# #
# #
# # # Function to load custom YOLOv5 model and perform object detection
# # def detect_objects(image_path: str, custom_weights_path: str, conf_thres: float = 0.25):
# #     # Load your custom YOLOv5 model
# #     model = torch.hub.load("ultralytics/yolov5", "custom", path=custom_weights_path)
# #
# #     # Load the image
# #     img = Image.open(image_path)
# #
# #     # Perform object detection
# #     results = model(img, size=640)  # Adjust size if necessary
# #
# #     # Extract bounding boxes and class predictions
# #     bboxes = results.xyxy[0].cpu().numpy()  # Convert bounding boxes to NumPy array
# #
# #     # Convert image to RGB mode
# #     img = img.convert("RGB")
# #
# #     # Convert PIL image to OpenCV format (numpy array)
# #     img_np = np.array(img)
# #
# #     # Loop through the detected objects
# #     for bbox in bboxes:
# #         x1, y1, x2, y2, conf, class_id = bbox
# #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #
# #         # Calculate the width and height of the bounding box
# #         width = x2 - x1
# #         height = y2 - y1
# #
# #         # Draw bounding box
# #         cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #
# #         # Display the width and height of the bounding box
# #         cv2.putText(img_np, f'Width: {width}, Height: {height}', (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# #
# #     # Save the annotated image with detections
# #     annotated_image_path = "annotated_" + os.path.basename(image_path)
# #     cv2.imwrite(annotated_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
# #
# #     return annotated_image_path
# #
# #
# # # Load the image
# # image_path = r'C:\Wall measurement new\WCL\Images\wall42.jpg'
# #
# # # Load your custom YOLOv5 model weights
# # custom_weights_path = r'C:\Wall measurement new\WCL\runs\train\exp10\weights\last.pt'
# #
# # # Perform object detection and measure object size
# # annotated_image_path = detect_objects(image_path, custom_weights_path)
# #
# # # Display the annotated image
# # img = cv2.imread(annotated_image_path)
# # # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from PIL import Image
# import torch
# import os
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
# import uvicorn
# app = FastAPI()

# custom_weights_path = 'last 1.pt'
# image_path = "images/"
# conf_thres = 0.25

# @app.post("/upload/")
# # Function to load custom YOLOv5 model and perform object detection
# def detect_objects(image_path: UploadFile = File(...)):
#     # Load your custom YOLOv5 model
#     model = torch.hub.load("ultralytics/yolov5", "custom", path=custom_weights_path)

#     # Load the image
#     img = Image.open(image_path)

#     # Perform object detection
#     results = model(img, size=640)  # Adjust size if necessary

#     # Extract bounding boxes, class predictions, and corresponding labels
#     bboxes = results.xyxy[0].cpu().numpy()  # Convert bounding boxes to NumPy array
#     labels = results.names[0]  # Extract class labels from the model

#     # Convert image to RGB mode
#     img = img.convert("RGB")

#     # Convert PIL image to OpenCV format (numpy array)
#     img_np = np.array(img)

#     # Loop through the detected objects
#     for bbox in bboxes:
#         x1, y1, x2, y2, conf, class_id = bbox
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#         # Calculate the width and height of the bounding box
#         width = x2 - x1
#         height = y2 - y1
#         # print(f"width : {width} and height : {height}")
#         # Draw bounding box
#         cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Display the width and height of the bounding box
#         cv2.putText(img_np, f'Width: {width}, Height: {height}', (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         # Add wall label
#         label = labels[int(class_id)]
#         cv2.putText(img_np, label, (x1, y2 + 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # Save the annotated image with detections
#     annotated_image_path = "annotated_" + os.path.basename(image_path)
#     cv2.imwrite(annotated_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
#     annotated_image_path = detect_objects(annotated_image_path, custom_weights_path)
#     img = cv2.imread(annotated_image_path)
#     # return annotated_image_path
#     return {"width" : width , "height" : height}


# # Load the image
# # image_path = r'C:\Wall measurement new\WCL\Images\wall25.jpg'
# # image_path = "wall11.jpg" 
# # Perform object detection and measure object size
# # annotated_image_path = detect_objects(img_final_path, custom_weights_path)

# # Display the annotated image
# # img = cv2.imread(annotated_image_path)
# # cv2.imshow('image', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()





# # @app.post("/upload/")
# # async def upload_image(image: UploadFile = File(...)):
# #     img_final_path = image_path +image.filename

# #     # with open(img_final_path, "wb") as buffer:
# #     #     buffer.write(image.file.read())
# #         # result = detect_objects(img_final_path, custom_weights_path, 0.25)
# #     result = detect_objects(img_final_path, custom_weights_path, 0.25)
# #     # return {"filename": image.filename,"status":"successful"}/
# #     return {result}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8002)


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

@app.post("/upload/")
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
