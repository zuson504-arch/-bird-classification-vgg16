import os
import time
import json
import numpy as np
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# INIT FLASK
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "model/bird_vgg16.keras"
model = load_model(MODEL_PATH)

# =========================
# LOAD CLASS NAMES
# =========================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

class_names = {int(k): v for k, v in class_names.items()}

IMG_SIZE = (224, 224)

# =========================
# HOME
# =========================


@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICT PAGE
# =========================
@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


# =========================
# PREDICT PROCESS
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return redirect("/predict")

    file = request.files["image"]

    if file.filename == "":
        return redirect("/predict")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image_path = filepath.replace("\\", "/")

    # timer
    start_time = time.time()

    # preprocess
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # prediction
    prediction = model.predict(img_array)[0]

    # TOP 3
    top3_idx = prediction.argsort()[-3:][::-1]

    top3 = []
    for i in top3_idx:
        top3.append({
            "label": class_names[i],
            "confidence": float(prediction[i]) * 100
        })

    result_label = top3[0]["label"]
    confidence = top3[0]["confidence"]

    # time
    end_time = time.time()
    inference_time = round(end_time - start_time, 2)

    return render_template(
        "result.html",
        prediction=result_label,
        confidence=round(confidence, 2),
        image_path=image_path,
        inference_time=inference_time,
        top3=top3
    )


# =========================
# ABOUT
# =========================
@app.route("/about")
def about():
    return render_template("about.html")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
