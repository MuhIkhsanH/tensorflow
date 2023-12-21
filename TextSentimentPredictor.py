import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Prepare Data
texts = ["This is an example of a positive sentence.",
         "I don't like this product.",
         "This product is really good.",
         "Very disappointed with customer service.",
         "I'm happy with my purchase.",
         "This product is very bad"
         ]
labels = [1, 0, 1, 0, 1,0]

labels = np.array(labels)  # Convert labels to a NumPy array

# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(text.split()) for text in texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Build the Model
model = Sequential([
    Embedding(vocab_size, 8, input_length=max_length),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(padded_sequences, labels, epochs=10, verbose=2)

# Prediction
new_text = ["This is bad"]
new_sequence = tokenizer.texts_to_sequences(new_text)
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_length, padding='post')

prediction = model.predict(new_padded_sequence)
print("Sentiment Prediction:", "Positive" if prediction[0][0] > 0.5 else "Negative")
