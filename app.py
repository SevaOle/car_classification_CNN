from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import pandas as pd
from PIL import Image

from flask import Flask, request, render_template_string
from tensorflow import keras


MODEL_PATH = Path("car_make_cnn.keras")
CSV_PATH = Path("split.csv")

IMG_SIZE = (224, 224)
TARGET_COLUMN = "make_id"
CLASS_NAME_COLUMN = "make"

app = Flask(__name__)

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

print("Loading class names...")
df = pd.read_csv(CSV_PATH)

class_names = (
    df[[TARGET_COLUMN, CLASS_NAME_COLUMN]]
    .drop_duplicates()
    .sort_values(TARGET_COLUMN)[CLASS_NAME_COLUMN]
    .tolist()
)

print("Classes:", class_names)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Make Recognition CNN</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 850px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }

        h1 {
            margin-top: 0;
        }

        img {
            max-width: 450px;
            max-height: 350px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .prediction {
            margin-top: 20px;
        }

        .bar-container {
            background: #ddd;
            border-radius: 8px;
            overflow: hidden;
            height: 22px;
            margin-bottom: 12px;
        }

        .bar {
            background: #4a90e2;
            height: 100%;
            color: white;
            text-align: right;
            padding-right: 8px;
            line-height: 22px;
            box-sizing: border-box;
            font-size: 13px;
        }

        .label {
            font-weight: bold;
            margin-bottom: 4px;
        }

        input[type="file"] {
            margin: 15px 0;
        }

        button {
            padding: 10px 18px;
            border: none;
            border-radius: 8px;
            background: #222;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #444;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Car Make Recognition CNN</h1>
        <p>Upload a car image and the trained CNN will predict the most likely make.</p>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <br>
            <button type="submit">Predict</button>
        </form>

        {% if image_data %}
            <h2>Uploaded Image</h2>
            <img src="data:image/jpeg;base64,{{ image_data }}">
        {% endif %}

        {% if predictions %}
            <div class="prediction">
                <h2>Top 5 Predictions</h2>

                {% for pred in predictions %}
                    <div class="label">
                        {{ pred.label }} — {{ pred.percent }}%
                    </div>
                    <div class="bar-container">
                        <div class="bar" style="width: {{ pred.percent }}%;">
                            {{ pred.percent }}%
                        </div>
                    </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def image_to_base64(image_file):
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_file.seek(0)

    return encoded


@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    image_data = None

    if request.method == "POST":
        uploaded_file = request.files["image"]

        image_data = image_to_base64(uploaded_file)

        image_array = preprocess_image(uploaded_file)

        probabilities = model.predict(image_array, verbose=0)[0]

        top_5_indexes = np.argsort(probabilities)[-5:][::-1]

        predictions = []

        for index in top_5_indexes:
            label = class_names[index]
            percent = round(float(probabilities[index]) * 100, 2)

            predictions.append({
                "label": label,
                "percent": percent
            })

    return render_template_string(
        HTML_PAGE,
        predictions=predictions,
        image_data=image_data
    )


if __name__ == "__main__":
    app.run(debug=True)