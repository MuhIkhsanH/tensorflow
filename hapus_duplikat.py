import pandas as pd

# Data training
texts = ["Saya suka produk ini.",
         "Saya tidak suka produk ini",
         "Saya menyayangi produk ini.",
         "Saya sangat membenci produk ini.",
         "Saya senang dengan produk ini.",
         "Produk ini sangat jelek.",
         "Saya benci produk ini.",
         "Produk ini saya benci.",
         "Saya tidak benci produk ini.",
         "Aku benar-benar menyukai produk ini!",
         "Saya sangat tidak menyukai produk ini!",
         "Saya tidak suka dengan produk ini.",
         "Saya suka dengan produk ini.",
         "Tidak suka dengan produk ini.",
         "Suka dengan produk ini."
         ]

labels = [1,0,1,0,1,0,0,0,1,1,0,0,1,0,1]

# Membuat DataFrame dari data teks dan label
df = pd.DataFrame({'text': texts, 'label': labels})

# Menambahkan kolom 'original_index' untuk menyimpan indeks asli
df['original_index'] = df.index

# Cetak teks sebelum penghapusan duplikat
print("Original Texts:")
for idx, text in enumerate(df['text']):
    print(f"{idx} {text}")

# Menghapus duplikat berdasarkan kolom 'text'
df_cleaned = df.drop_duplicates(subset=['text'])

# Menyimpan data yang sudah dibersihkan
cleaned_texts = df_cleaned['text'].tolist()

# Cetak teks setelah penghapusan duplikat
print("\nCleaned Texts:")
for idx, text in enumerate(cleaned_texts):
    print(f"{idx} {text}")

# Cetak indeks asli dari data yang terduplikat
print("\nOriginal Indices of Duplicates:")
for idx, original_index in enumerate(df[df.duplicated(subset=['text'], keep=False)]['original_index']):
    print(f"Original Index {idx}: {original_index}")
