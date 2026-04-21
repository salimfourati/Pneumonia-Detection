from pathlib import Path
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

try:
    model = load_model(MODEL_PATH)
    print(f"Modèle chargé depuis : {MODEL_PATH}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Le modèle n'a pas pu être chargé.", 500

    if 'file' not in request.files:
        return "Aucun fichier sélectionné.", 400

    file = request.files['file']
    if file.filename == '':
        return "Nom de fichier vide.", 400

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(file_path)

    try:
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = model.predict(img_array, verbose=0)[0]

        classes = ['Normal', 'Pneumonie']
        predicted_class = int(np.argmax(prediction))
        predicted_label = classes[predicted_class]
        probability = round(float(prediction[predicted_class] * 100), 2)

        image_url = url_for('static', filename=f'uploads/{filename}')

        return render_template(
            'result.html',
            result=predicted_label,
            probability=probability,
            image_url=image_url
        )

    except Exception as e:
        return f"Erreur lors de la prédiction : {e}", 500


if __name__ == '__main__':
    app.run(debug=True)