import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Load model terbaru ---
def load_model():
    model_dir = "./models"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Folder './models' tidak ditemukan.")

    # Ambil file yang berformat .h5 dan awalan angka
    versions = [int(f.split(".")[0]) for f in os.listdir(model_dir) if f.endswith(".h5") and f.split(".")[0].isdigit()]
    
    if not versions:
        raise FileNotFoundError("Tidak ada file model '.h5' dalam folder './models'. Contoh: './models/1.h5'")

    latest_version = str(max(versions))
    model_path = os.path.join(model_dir, latest_version + ".h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di path: {model_path}")

    return tf.keras.models.load_model(model_path)

# --- Fungsi prediksi ---
def predict(model, image, class_names, image_size=(256, 256)):
    img = image.resize(image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# --- Daftar kelas ---
class_names = ['Early blight', 'Late blight', 'Healthy']  # Sesuaikan dengan datasetmu

# --- UI Streamlit ---
st.set_page_config(page_title="Deteksi Penyakit Tanaman Kentang", layout="centered")
st.title("🌿 Deteksi Penyakit Daun Kentang")
st.write("Upload gambar daun, dan model akan memprediksi penyakitnya.")

uploaded_file = st.file_uploader("Upload gambar daun (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_container_width=True)  # ← ini sudah benar sekarang

    model = load_model()

    st.write("🔍 Memprediksi...")
    predicted_class, confidence = predict(model, image, class_names)

    st.success(f"**Prediksi:** {predicted_class}")
    st.info(f"**Akurasi model:** {confidence}%")
