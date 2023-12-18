from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="assets/modelv3.tflite")
interpreter.allocate_tensors()

# Load labels
with open("assets/labels.txt", "r") as file:
    labels = file.read().splitlines()

# Serve HTML page with Bootstrap styles
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Image Classification</title>
        <!-- Bootstrap CSS -->
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="container mt-4">
        <h1 class="mb-4">Image Classification</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input class="form-control mb-3" type="file" name="file">
            <button class="btn btn-primary" type="submit">Upload & Predict</button>
        </form>
    </body>
    </html>
    """
    return content

# Define endpoint for image classification and display result as HTML
@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_resized = image.resize((224, 224))  # Adjust size as per your model's input requirements
    image_array = np.array(image_resized, dtype=np.float32)
    image_array /= 255.0  # Normalize if needed

    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], [image_array])
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(output_data[0])

    # Get prediction confidence
    prediction_confidence = output_data[0][predicted_class_index]

    if prediction_confidence < 0.8:
        predicted_label = "Unknown Fruit"
    else:
        predicted_label = labels[predicted_class_index]

    # Convert image to base64 for displaying in HTML
    buffered = BytesIO()
    image_resized.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    result_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Prediction Result</title>
        <!-- Bootstrap CSS -->
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="container mt-4">
        <h1>Prediction Result</h1>
        <div class="row">
            <div class="col-md-6">
                <h3>Uploaded Image</h3>
                <img src="data:image/jpeg;base64,{img_str}" class="img-fluid" alt="Uploaded Image">
            </div>
            <div class="col-md-6">
                <h3>Prediction</h3>
                <div class="alert alert-success" role="alert">
                    Predicted Label: {predicted_label}
                </div>
                <div class="alert alert-info" role="alert">
                    Confidence: {prediction_confidence:.2f}
                </div>
                <p><a href="/">Back to Upload</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return result_html
