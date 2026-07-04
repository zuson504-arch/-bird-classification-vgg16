import os
import time
import json
import numpy as np
import gdown
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
os.makedirs("model", exist_ok=True)

# =========================
# GOOGLE DRIVE MODEL AUTO DOWNLOAD
# =========================
MODEL_PATH = "model/bird_vgg16.keras"
FILE_ID = "12UWg00lCeotFFjpvMZm07-K1wRdr45c2"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)

# =========================
# LOAD CLASS NAMES
# =========================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

class_names = {int(k): v for k, v in class_names.items()}

IMG_SIZE = (224, 224)

# =========================
# ROUTES
# =========================


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")


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

    start_time = time.time()

    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0]

    top3_idx = prediction.argsort()[-3:][::-1]

    top3 = []
    for i in top3_idx:
        top3.append({
            "label": class_names[i],
            "confidence": float(prediction[i]) * 100
        })

    result_label = top3[0]["label"]
    confidence = top3[0]["confidence"]

    inference_time = round(time.time() - start_time, 2)

    return render_template(
        "result.html",
        prediction=result_label,
        confidence=round(confidence, 2),
        image_path=image_path,
        inference_time=inference_time,
        top3=top3
    )


@app.route("/about")
def about():
    return render_template("about.html")


# =========================
# RUN (RAILWAY READY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
