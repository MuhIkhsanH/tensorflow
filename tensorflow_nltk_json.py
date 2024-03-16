import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

# Fungsi untuk membaca dataset dari file JSON
def read_dataset_from_json(json_file):
    with open(json_file, 'r') as file:
        dataset = json.load(file)
    questions = [item['question'] for item in dataset]
    answers = [item['answer'] for item in dataset]
    return questions, answers

# Fungsi untuk melatih model chatbot
def train_chatbot_model(questions, answers):
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

    # Membangun model LSTM sederhana dengan dropout
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),  # Dropout untuk mencegah overfitting
        tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
    ])

    # Kompilasi model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Melatih model
    model.fit(question_sequences, answer_sequences, epochs=200, verbose=2)

    return model, tokenizer, max_length

# Fungsi untuk menghasilkan jawaban berdasarkan pertanyaan
def generate_response(question, model, tokenizer, max_length):
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = pad_sequences(question_seq, maxlen=max_length, padding='post')
    predicted_answer = model.predict(question_seq)
    predicted_answer = np.argmax(predicted_answer, axis=-1)
    predicted_answer = tokenizer.sequences_to_texts(predicted_answer)
    return predicted_answer[0]

# Baca dataset dari file JSON
questions, answers = read_dataset_from_json('dataset.json')

# Latih model chatbot
model, tokenizer, max_length = train_chatbot_model(questions, answers)

# Uji model chatbot
while True:
    user_input = input("You: ")
    response = generate_response(user_input, model, tokenizer, max_length)
    print("Bot:", response)
