import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load dataset from JSON file
with open('dataset.json', 'r') as file:
    dataset = json.load(file)

# Extract questions and answers from dataset
questions = [item['question'] for item in dataset]
answers = [item['answer'] for item in dataset]

# Tokenize questions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
question_sequences = tokenizer.texts_to_sequences(questions)
max_length = max([len(seq) for seq in question_sequences])

# Pad sequences
question_sequences_padded = pad_sequences(question_sequences, maxlen=max_length, padding='post')

# Define model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length-1),
    LSTM(512),
    Dense(vocab_size, activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Encode answers using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(answers)
encoded_answers = label_encoder.transform(answers)

# Train model
model.fit(question_sequences_padded[:, :-1], encoded_answers, epochs=50, verbose=1)

# Save model
model.save('chatbot_model')

# Function to generate response
def generate_response(input_question):
    question_seq = tokenizer.texts_to_sequences([input_question])
    question_seq = pad_sequences(question_seq, maxlen=max_length-1, padding='post')
    predicted_answer = model.predict(question_seq)
    predicted_answer_index = np.argmax(predicted_answer)
    predicted_answer = label_encoder.inverse_transform([predicted_answer_index])[0]
    return predicted_answer

# Example usage
input_question = "how are you"
response = generate_response(input_question)
print(response)
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load dataset from JSON file
with open('dataset.json', 'r') as file:
    dataset = json.load(file)

# Extract questions and answers from dataset
questions = [item['question'] for item in dataset]
answers = [item['answer'] for item in dataset]

# Tokenize questions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
question_sequences = tokenizer.texts_to_sequences(questions)
max_length = max([len(seq) for seq in question_sequences])

# Pad sequences
question_sequences_padded = pad_sequences(question_sequences, maxlen=max_length, padding='post')

# Define model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length-1),
    LSTM(512),
    Dense(vocab_size, activation='softmax')
])

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Encode answers using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(answers)
encoded_answers = label_encoder.transform(answers)

# Train model
model.fit(question_sequences_padded[:, :-1], encoded_answers, epochs=50, verbose=1)

# Save model
model.save('chatbot_model')

# Function to generate response
def generate_response(input_question):
    question_seq = tokenizer.texts_to_sequences([input_question])
    question_seq = pad_sequences(question_seq, maxlen=max_length-1, padding='post')
    predicted_answer = model.predict(question_seq)
    predicted_answer_index = np.argmax(predicted_answer)
    predicted_answer = label_encoder.inverse_transform([predicted_answer_index])[0]
    return predicted_answer

# Example usage
input_question = "how are you"
response = generate_response(input_question)
print(response)
