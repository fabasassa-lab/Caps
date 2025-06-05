from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from PIL import Image
from flask_cors import CORS
import time
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model


app = Flask(__name__, static_folder='images', static_url_path='/images')
CORS(app)
model = load_model('Model/densenet201.keras')

UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

label_map = {
    0: 'Anthracnose',
    1: 'Bacterial Canker',
    2: 'Cutting Weevil',
    3: 'Die Back',
    4: 'Gall Midge',
    5: 'Healthy',
    6: 'Powdery Mildew',
    7: 'Sooty Mould'
}

recommendation_map = {
    'Anthracnose': 'Gunakan fungisida berbahan aktif tembaga dan tambahkan pupuk AB1.',
    'Bacterial Canker': 'Potong bagian yang terinfeksi dan gunakan bakterisida alami.',
    'Cutting Weevil': 'Gunakan insektisida kontak dan periksa kebersihan lingkungan sekitar tanaman.',
    'Die Back': 'Lakukan pemangkasan daun mati dan beri pupuk tinggi kalium.',
    'Gall Midge': 'Semprotkan insektisida sistemik dan kontrol kelembaban tanah.',
    'Healthy': 'Tanaman sehat! Lanjutkan pemupukan dan penyiraman rutin.',
    'Powdery Mildew': 'Semprot dengan larutan belerang atau fungisida sistemik.',
    'Sooty Mould': 'Hilangkan sumber kutu putih dan cuci daun dengan sabun hortikultura.'
}

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))  # ukuran untuk CNN
    img_array = img_to_array(img)  # (224, 224, 3)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    print("MASUK KE ROUTE POST")
    imagefile = request.files['imagefile']
    print("Image file diterima:", imagefile.filename)

    if imagefile.filename == '':
        return jsonify({
            "success": False,
            "error": "Tidak ada file yang diunggah."
        })

    filename = f"{int(time.time())}_{secure_filename(imagefile.filename)}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    imagefile.save(image_path)

    image = extract_features(image_path)
    print("Image shape:", image.shape)

    prediction = model.predict(image)
    print("Raw prediction:", prediction)

    predicted_label = np.argmax(prediction)
    label_name = label_map[predicted_label]
    confidence = prediction[0][predicted_label] * 100
    final_result = f"{label_name} ({confidence:.2f}%)"
    recommendation = recommendation_map.get(label_name, "Tidak ada rekomendasi khusus.")

    return jsonify({
        "success": True,
        "prediction": final_result,
        "recommendation": recommendation,
        "image_url": f"/images/{filename}"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
