from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="assets/model.tflite")
interpreter.allocate_tensors()

# Load labels
with open("assets/labels.txt", "r") as file:
    labels = file.read().splitlines()

# Serve HTML page
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
    <head>
        <title>Image Classification</title>
    </head>
    <body>
        <h1>Image Classification</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """
    return content

# Define endpoint for image classification
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))  # Adjust size as per your model's input requirements
    image = np.array(image, dtype=np.float32)
    image = image / 255.0  # Normalize if needed

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], [image])
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(output_data[0])
    predicted_label = labels[predicted_class_index]

    return {"predicted_label": predicted_label}
