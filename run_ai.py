import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
loaded_model = load_model("/content/sentiment_model.h5")

# Input text for prediction
new_text = ["Saya tidak suka produk ini."]
new_sequence = tokenizer.texts_to_sequences(new_text)
new_padded_sequence = pad_sequences(new_sequence, maxlen=max_length, padding='post')

# Make prediction
prediction = loaded_model.predict(new_padded_sequence)
print("Sentiment Prediction:", "Positive" if prediction[0][0] > 0.5 else "Negative")
