import pandas as pd
import re
import nltk
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding,SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load dataset
df = pd.read_csv("data\labeled_data_full.csv")

# Unduh stopwords untuk bahasa Indonesia (dengan error handling)
try:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("indonesian"))
except:
    stop_words = set()

# Fungsi untuk membersihkan teks
def clean_text(text):
    if isinstance(text, str): # Pastikan teks bukan NaN atau None

        text = text.lower() # Ubah ke huruf kecil
        text = re.sub(r"[^\w\s]", "", text) # Hapus tanda baca
        text = " ".join([word for word in text.split() if word
        not in stop_words]) # Hapus stopword
        return text
    return "" # Jika teks kosong atau NaN, return string kosong

# Terapkan pembersihan teks
df["clean_text"] = df["komentar"].astype(str).apply(clean_text)

# Pastikan label bertipe integer
df["label"] = df["label"].astype(int)

# Cek jumlah data per kelas
print("Distribusi Kelas:\n", df["label"].value_counts())

# Tokenisasi teks
max_words = 5000 # Jumlah maksimal kata unik dalam kamus
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_text"])
X = tokenizer.texts_to_sequences(df["clean_text"])
X = pad_sequences(X, maxlen=100, padding="post", truncating="post")

# Ambil label sebagai target
y = df["label"].values

# Bagi data menjadi training dan testing (stratify untuk keseimbangan data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Model LSTM (dengan Bidirectional)
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128,
    input_length=100),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(100, dropout=0.2,
    recurrent_dropout=0.2)), # Bidirectional LSTM
    Dense(3, activation='softmax') # Output 2 kelas (Negatif, Positif)
])

# Kompilasi model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training model
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

# Prediksi
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluasi model dengan zero_division=1
print(classification_report(y_test, y_pred_classes, zero_division=1))

# Simpan model ke file .h5
model.save("model_sentimen.h5")

# Simpan tokenizer ke file .pkl
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
