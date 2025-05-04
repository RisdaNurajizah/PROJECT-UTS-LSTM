from flask import Flask, render_template, request
import numpy as np
import re
import nltk
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan tokenizer
model = load_model("model_sentimen.h5")  # Model hasil training disimpan sebagai .h5
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Unduh stopwords jika belum ada
try:
    nltk.download("stopwords")
except:
    pass

stop_words = set(stopwords.words("indonesian"))

# Fungsi untuk membersihkan teks (sama seperti di training)
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text
    return ""

# Routing untuk halaman utama
@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    img_base64 = None  # Inisialisasi grafik sebagai None
    if request.method == "POST":
        komentar = request.form["komentar"]
        cleaned = clean_text(komentar)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
        prediction = model.predict(padded)
        
        # Dapatkan probabilitas untuk setiap kelas (Negatif, Netral, Positif)
        sentiment_probs = prediction[0]  # Ambil probabilitas dari prediksi pertama
        sentiment_labels = ["Negatif", "Netral", "Positif"]
        
        # Cari label dengan probabilitas tertinggi untuk hasil prediksi
        label = np.argmax(sentiment_probs)

        # Ubah label ke bentuk teks (opsional tergantung dataset)
        hasil = sentiment_labels[label]

        # Membuat diagram batang untuk distribusi sentimen
        plt.figure(figsize=(6, 4))
        plt.bar(sentiment_labels, sentiment_probs, color=["red", "gray", "green"])
        plt.title("Distribusi Sentimen")
        plt.ylabel("Probabilitas")

        # Simpan grafik sebagai file PNG
        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("index.html", hasil=hasil, graph=img_base64)

# Jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)
