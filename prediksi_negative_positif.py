import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# Data training
texts = ["Saya suka produk ini.",
         "Saya tidak suka produk ini",
         "Saya menyayangi produk ini.",
         "Saya sangat membenci produk ini.",
         "Saya senang dengan produk ini.",
         "Produk ini sangat jelek.",
         "Saya benci produk ini.",
         "Saya tidak benci produk ini.",
         "Produk ini saya benci.",
         "Saya tidak benci produk ini.",
         "Aku benar-benar menyukai produk ini!",
         "Saya sangat tidak menyukai produk ini!",

         ]
labels = [1, 0, 1, 0, 1, 0,0,1,0,1,1,0]

# Tokenization and Padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(text.split()) for text in texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to NumPy array
labels = np.array(labels)

# Build the model
model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callback for plotting
class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))

        # Plotting
        if epoch % 5 == 0:  # Plot setiap 5 epoch
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.losses, label='loss')
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(self.accuracies, label='accuracy')
            plt.title('Training Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.show()

# Train the model with the callback
model.fit(padded_sequences, labels, epochs=20, verbose=2, callbacks=[PlotLosses()])

# Prediction
new_text = ["Saya menyukai produk ini."]
new_sequence = tokenizer.texts_to_sequences(new_text)
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_length, padding='post')

prediction = model.predict(new_padded_sequence)
print("Sentiment Prediction:", "Positive" if prediction[0][0] > 0.5 else "Negative")
