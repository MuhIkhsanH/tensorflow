import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load dataset from JSON file
with open('dataset.json', 'r') as file:
    dataset = json.load(file)

# Extract questions and answers from dataset
questions = [item['question'] for item in dataset]
answers = [item['answer'] for item in dataset]

# Tokenize questions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

# Pad sequences
question_sequences = tokenizer.texts_to_sequences(questions)
max_length = max([len(seq) for seq in question_sequences])
question_sequences_padded = pad_sequences(question_sequences, maxlen=max_length, padding='post')

# Load model
model = load_model('chatbot_model')

# Encode answers using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(answers)

# Function to generate response
def generate_response(input_question, tokenizer, max_length, model, label_encoder):
    question_seq = tokenizer.texts_to_sequences([input_question])
    question_seq = pad_sequences(question_seq, maxlen=max_length-1, padding='post')
    predicted_answer = model.predict(question_seq)
    predicted_answer_index = np.argmax(predicted_answer)
    predicted_answer = label_encoder.inverse_transform([predicted_answer_index])[0]
    return predicted_answer

# Example usage
input_question = "hello bot"
response = generate_response(input_question, tokenizer, max_length, model, label_encoder)
print("Response:", response)
