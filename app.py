import io

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("modelo.h5")


def preparar_imagen(archivo):
    """Mismo preprocesamiento que en el notebook de Colab."""
    img = Image.open(io.BytesIO(archivo.read())).convert("L") 
    img = ImageOps.invert(img)                                  
    img = img.resize((28, 28))                                 
    arr = np.array(img) / 255.0                                
    arr = np.expand_dims(arr, axis=0)                           

    if len(model.input_shape) == 4:
        arr = np.expand_dims(arr, axis=-1)
    return arr


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/predecir", methods=["POST"])
def predecir():
    if "imagen" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    try:
        entrada = preparar_imagen(request.files["imagen"])
        predicciones = model.predict(entrada)
        numero = int(np.argmax(predicciones))
        confianza = float(np.max(predicciones)) * 100
        return jsonify({"numero": numero, "confianza": round(confianza, 2)})
    except Exception as e:
        return jsonify({"error": f"No se pudo procesar la imagen: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
