from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import shutil
from make_prediction import object_detection  # Importing your object detection function

app = FastAPI()

# Set the path to Tesseract in your system
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Define the endpoint for image upload and processing
@app.post("/upload-and-ocr/")
async def upload_and_ocr(file: UploadFile = File(...)) -> JSONResponse:
    # Save the uploaded image to the specified path
    path = f'./test_images/{file.filename}'
    with open(path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Ensure the file is closed after saving
    await file.close()
    
    # Call the object_detection method with the saved image path
    # The object_detection function is assumed to be defined correctly in the make_prediction module
    image, cods = object_detection(path)
    
    # Perform OCR on the region of interest (ROI) from the object detection result
    img = np.array(load_img(path))
    xmin, xmax, ymin, ymax = cods[0]
    roi = img[ymin:ymax, xmin:xmax]
    plt.figure(figsize=(10,8))
    plt.imshow(roi)
    plt.pause(3)  # Display the figure for 5 seconds
    plt.close()  # Close the figure programmatically
    
    # Extract text from image using Tesseract OCR
    text = pytesseract.image_to_string(roi)
    if not text.strip():  # This checks if the text is empty or contains only whitespace
        text = "The image is not clear enough for text recognition."
    else:
        # Proceed with your logic if text is found
        print(text)
    # Return the OCR text as the response
    return JSONResponse(status_code=200, content={"extracted_text": text})

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
