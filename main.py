import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

reddit_data = np.load("reddit.npy")
hackernews_data = np.load("hackernews.npy")
youtube_data = np.load("youtube.npy")

samples = 800000
max_comment_length = 200
word_amount = 100000
embedding_dim = 128

x_train = np.zeros(3 * samples)
x_val = np.zeros(3 * samples)
x_test = np.zeros(3 * samples)

tokenizer = Tokenizer(num_words=word_amount)
tokenizer.fit_on_texts(x_train) # må kanskje skrive list(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=max_comment_length)
x_val = pad_sequences(x_val, maxlen=max_comment_length)
x_test = pad_sequences(x_test, maxlen=max_comment_length)

model = Sequential()
model.add(Embedding(word_amount, 128, input_length=max_comment_length))
model.add(LSTM(128))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", "precision", "recall"])

model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
evaluation = model.evaluate(x_test, y_test, verbose=0)
print(evaluation)