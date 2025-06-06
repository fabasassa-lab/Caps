import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io

# --- Setup Page ---
st.set_page_config(
    page_title="MANGALYZE - Analisis Daun Mangga",
    layout="wide",
    page_icon="🍃"
)

# --- Load TFLite Model ---
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="Model/densenet201.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
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

# --- Fungsi Preprocessing untuk TFLite ---
def preprocess(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    # Scaling pixel values to [-1,1] sesuai DenseNet
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
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
    
    image = None
    if example:
        # Pastikan contoh gambar ada di folder 'sample_images/sample_healthy.jpg'
        try:
            with open("sample_images/sample_healthy.jpg", "rb") as f:
                image = Image.open(io.BytesIO(f.read()))
        except FileNotFoundError:
            st.error("File contoh gambar tidak ditemukan. Upload gambar manual.")
    elif uploaded_file:
        image = Image.open(uploaded_file)

with col2:
    if image:
        st.image(image, caption="Gambar Daun", use_column_width=True)
    else:
        st.info("Unggah gambar daun mangga untuk mulai analisis.")

st.markdown("")

# --- Prediksi dan Output ---
if image and st.button("🔍 Analisis Daun"):
    with st.spinner("Sedang memproses..."):
        try:
            input_data = preprocess(image)
            # Pastikan tipe input sesuai input_details
            if input_details[0]['dtype'] == np.uint8:
                input_data = ((input_data + 1) * 127.5).astype(np.uint8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data)
            confidence = output_data[0][predicted_label] * 100
            label_name = label_map.get(predicted_label, "Unknown")
            recommendation = recommendation_map.get(label_name, "Tidak ada rekomendasi khusus.")
            
            st.success("✅ Analisis Selesai!")
            st.markdown(f"<h3 style='color:#4CAF50;'>Hasil Prediksi: <b>{label_name}</b> ({confidence:.2f}%)</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#f0f9f0; padding:10px; border-radius:10px'><b>📌 Rekomendasi:</b> {recommendation}</div>", unsafe_allow_html=True)
        except RuntimeError as e:
            st.error(f"Runtime error saat inferensi model: {e}")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

elif not image and st.button("🔍 Analisis Daun"):
    st.warning("Silakan unggah gambar terlebih dahulu.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>© 2025 MANGALYZE | Didukung oleh Deep Learning dan Cinta Tani</div>",
    unsafe_allow_html=True
)
