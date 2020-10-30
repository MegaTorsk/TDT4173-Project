import numpy as np
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Attention, Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report
#import tensorflowjs as tfjs


max_comment_length = 100
word_amount = 10000
embedding_dim = 128

reddit_file0 = open("formated_data/reddit0.pickle", "rb")
reddit_data0 = pickle.load(reddit_file0)
reddit_file0.close()
reddit_file1 = open("formated_data/reddit1.pickle", "rb")
reddit_data1 = pickle.load(reddit_file1)
reddit_file1.close()
reddit_file2 = open("formated_data/reddit2.pickle", "rb")
reddit_data2 = pickle.load(reddit_file2)
reddit_file2.close()
reddit_file3 = open("formated_data/reddit3.pickle", "rb")
reddit_data3 = pickle.load(reddit_file3)
reddit_file3.close()

reddit_data = reddit_data0 + reddit_data1 + reddit_data2 + reddit_data3 # Combine lists

hackernews_file = open("formated_data/hacker_news.pickle", "rb")
hackernews_data = pickle.load(hackernews_file)
hackernews_file.close()

youtube_file = open("formated_data/youtube.pickle", "rb")
youtube_data = pickle.load(youtube_file)
youtube_file.close()

reddit_samples = len(reddit_data)
hackernews_samples = len(hackernews_data)
youtube_samples = len(youtube_data)
samples = reddit_samples + hackernews_samples + youtube_samples

reddit_labels = [0 for i in range(reddit_samples)]
hackernews_labels = [1 for i in range(hackernews_samples)]
youtube_labels = [2 for i in range(youtube_samples)]

data = reddit_data + hackernews_data + youtube_data
labels = reddit_labels + hackernews_labels + youtube_labels
data, labels = shuffle(data, labels)

x_train = data[0 : samples * 70 // 100]
x_val = data[samples * 70  // 100 : samples * 85 // 100]
x_test = data[samples * 85 // 100 :]

y_train = labels[0 : samples * 70 // 100]
y_val = labels[samples * 70  // 100 : samples * 85 // 100]
y_test = labels[samples * 85 // 100 :]

class_weights = class_weight.compute_class_weight("balanced", [0, 1, 2], y_train)
class_weights = dict(enumerate(class_weights))

tokenizer = Tokenizer(num_words=word_amount)
tokenizer.fit_on_texts(x_train)

f = open("tokenizer.txt", "w")
f.write(str(tokenizer.word_index))
f.close()

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=max_comment_length)
x_val = pad_sequences(x_val, maxlen=max_comment_length)
x_test = pad_sequences(x_test, maxlen=max_comment_length)

y_train = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_train)))
y_val = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_val)))
y_test = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_test)))


models = []

# Logistic regression
model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(Dense(3, activation="softmax"))
models.append(model)

# LSTM
model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(LSTM(embedding_dim))
model.add(Dense(3, activation="softmax"))
models.append(model)

# Bidirectional LSTM
model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(3, activation="softmax"))
models.append(model)

# GRU
model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(GRU(embedding_dim))
model.add(Dense(3, activation="softmax"))
models.append(model)

# Bidirectional GRU
model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(Bidirectional(GRU(embedding_dim)))
model.add(Dense(3, activation="softmax"))
models.append(model)

# Droput og regualrization fikser overfitting, men trengs forhåpentligvis ikke
# NumPy arrays er raskere og mer plasseffektive når elementene er tall. For stringer er de ikke det

for i in range(len(models)):
	model = models[i]
	model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
	model.fit(x_train, y_train, class_weight=class_weights, epochs=10, batch_size=128, verbose=1)
	#tfjs.converters.save_keras_model(model, "model" + str(i))

	f = open("model" + str(i) + ".txt")

	loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
	f.write("Validation: [loss: " + str(loss) + ", accuracy: " + str(accuracy) + "]")
	y_val_pred = model.predict(x_val)
	f.write(classification_report(np.argmax(y_val, axis=1), np.argmax(y_val_pred, axis=1), target_names=["Reddit", "Hacker News", "YouTube"]))

	loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
	f.write("Test: [loss: " + str(loss) + ", accuracy: " + str(accuracy) + "]")
	y_test_pred = model.predict(x_test)
	f.write(classification_report(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1), target_names=["Reddit", "Hacker News", "YouTube"]))

	f.close()
