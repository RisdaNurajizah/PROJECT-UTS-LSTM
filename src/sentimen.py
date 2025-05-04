import pandas as pd
import random

# Load file CSV
df = pd.read_csv("data/komentar_twitter.csv")

# Isi label_sentimen dengan acak (string)
labels = ["positif", "netral", "negatif"]
df["sentimen"] = [random.choice(labels) for _ in range(len(df))]

# Mapping label ke angka
label_mapping = {
    "positif": 1,
    "netral": 2,
    "negatif": 0
}
df["label"] = df["sentimen"].map(label_mapping)

# Simpan hasil
df.to_csv("data/labeled_data_full.csv", index=False)
print("Berhasil generate label dengan angka!")
