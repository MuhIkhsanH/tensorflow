import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

# Contoh dataset pertanyaan dan jawaban
questions = ["What is your name?", "How are you?", "What can you do?", "Who created you?"]
answers = ["My name is Chatbot.", "I'm fine, thank you!", "I can answer your questions.", "I was created by OpenAI."]

# Tokenisasi teks menggunakan NLTK
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# Mengonversi teks ke dalam sequences
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Padding sequences untuk memastikan ukuran input yang seragam
max_length = max(len(seq) for seq in question_sequences)
question_sequences = pad_sequences(question_sequences, maxlen=max_length, padding='post')
answer_sequences = pad_sequences(answer_sequences, maxlen=max_length, padding='post')

# Membangun model LSTM sederhana
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Kompilasi model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Melatih model
model.fit(question_sequences, answer_sequences, epochs=100, verbose=2)

# Fungsi untuk menghasilkan jawaban berdasarkan pertanyaan
def generate_response(question):
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = pad_sequences(question_seq, maxlen=max_length, padding='post')
    predicted_answer = model.predict(question_seq)
    predicted_answer = np.argmax(predicted_answer, axis=-1)
    predicted_answer = tokenizer.sequences_to_texts(predicted_answer)
    return predicted_answer[0]

# Uji model
while True:
    user_input = input("You: ")
    response = generate_response(user_input)
    print("Bot:", response)
