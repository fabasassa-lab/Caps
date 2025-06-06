import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input

# --- Setup Page ---
st.set_page_config(
    page_title="MANGALYZE - Analisis Daun Mangga",
    layout="wide",
    page_icon="🍃"
)

# --- Load TFLite Model ---
interpreter = tf.lite.Interpreter(model_path="Model/densenet201.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Label dan Rekomendasi ---
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
    'Anthracnose': 'Gunakan fungisida berbahan aktif (mankozeb, tembaga hidroksida, atau propineb) sesuai dosis anjuran.',
    'Bacterial Canker': 'Potong bagian yang terinfeksi dan gunakan bakterisida berbahan tembaga (copper-based).',
    'Cutting Weevil': 'Gunakan insektisida berbahan aktif (imidakloprid, lambda-cyhalothrin) dan periksa kebersihan lingkungan sekitar tanaman.',
    'Die Back': 'Lakukan pemangkasan daun mati dan semprotkan fungisida sistemik (benomil, karbendazim, tebuconazole).',
    'Gall Midge': 'Pangkas dan bakar daun/bunga yang terinfestasi dan aplikasikan insektisida sistematik (imidakloprid, abamektin, spinosad).',
    'Healthy': 'Tanaman sehat! Lanjutkan pemupukan dan penyiraman rutin.',
    'Powdery Mildew': 'Semprot dengan fungisida sistemik dan preventif (karathane, hexaconazole, sulfur, miklobutanil).',
    'Sooty Mould': 'Pangkas ranting yang terlalu rimbun dan semprot air sabun ringan atau campuran air + fungisida ringan.'
}

# --- Fungsi Ekstraksi Fitur ---
def extract_features(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# --- Header ---
st.markdown("<h1 style='text-align: center; color: green;'>🍃 MANGALYZE</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Sistem Deteksi Penyakit Daun Mangga Berbasis Deep Learning</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Layout Utama ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Unggah Gambar Daun")
    uploaded_file = st.file_uploader("Format gambar: JPG / PNG", type=["jpg", "jpeg", "png"])
    example = st.checkbox("Gunakan contoh gambar (Healthy)")
    if example:
        uploaded_file = "sample_healthy.jpg"  # Pastikan file contoh ini ada di direktori
        image = Image.open(uploaded_file)
    elif uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = None

with col2:
    if image:
        st.image(image, caption="Gambar Daun", use_column_width=True)
    else:
        st.info("Unggah gambar daun mangga untuk mulai analisis.")

st.markdown("")

# --- Prediksi dan Output ---
if image and st.button("🔍 Analisis Daun"):
    with st.spinner("Sedang memproses..."):
        features = extract_features(image)
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(prediction)
        label_name = label_map[predicted_label]
        confidence = prediction[0][predicted_label] * 100
        recommendation = recommendation_map.get(label_name, "Tidak ada rekomendasi khusus.")

    st.success("✅ Analisis Selesai!")
    st.markdown(f"<h3 style='color:#4CAF50;'>Hasil Prediksi: <b>{label_name}</b> ({confidence:.2f}%)</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#f0f9f0; padding:10px; border-radius:10px'><b>📌 Rekomendasi:</b> {recommendation}</div>", unsafe_allow_html=True)
elif not image and st.button("🔍 Analisis Daun"):
    st.warning("Silakan unggah gambar terlebih dahulu.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>© 2025 MANGALYZE | Didukung oleh Deep Learning dan Cinta Tani</div>",
    unsafe_allow_html=True
)
