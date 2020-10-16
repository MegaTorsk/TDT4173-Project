import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight

max_comment_length = 200
word_amount = 100000
embedding_dim = 128

#reddit_data = np.load("reddit.npy")
#reddit_data = np.array([])
hackernews_data = np.load("hacker_news_test.npy")
youtube_data = np.load("youtube_test.npy")

#reddit_samples = reddit_data.shape[0]
hackernews_samples = hackernews_data.shape[0]
youtube_samples = youtube_data.shape[0]
#samples = reddit_samples + hackernews_samples + youtube_samples
samples = hackernews_samples + youtube_samples

#reddit_labels = np.array([0 for i in range(reddit_samples)])
hackernews_labels = np.array([1 for i in range(hackernews_samples)])
youtube_labels = np.array([2 for i in range(youtube_samples)])

#data = np.concatenate((reddit_data, hackernews_data, youtube_data), axis=0)
data = np.concatenate((hackernews_data, youtube_data), axis=0)
#labels = np.concatenate((reddit_labels, hackernews_labels, youtube_labels), axis=0)
labels = np.concatenate((hackernews_labels, youtube_labels), axis=0)
data = np.stack((data, labels), axis=1)
np.random.shuffle(data)

x_train = data[0 : samples * 70 // 100, 0]
y_train = data[0 : samples * 70 // 100, 1]
x_val = data[samples * 70  // 100 : samples * 85 // 100, 0]
y_val = data[samples * 70  // 100 : samples * 85 // 100, 1]
x_test = data[samples * 85 // 100 :, 0]
y_test = data[samples * 85 // 100 :, 1]

#class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
#print(class_weights)

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
model.add(Dense(1, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

#model.fit(x_train, y_train, class_weight=class_weights, epochs=3, batch_size=128, verbose=0)
model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)
evaluation = model.evaluate(x_test, y_test, verbose=1)
print(evaluation)