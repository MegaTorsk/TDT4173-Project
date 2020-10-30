import numpy as np
import tensorflowjs as tfjs
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Attention, Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

max_comment_length = 200
word_amount = 10000
embedding_dim = 128

reddit_data = np.load("reddit.npy")
hackernews_data = np.load("hacker_news_test.npy")
youtube_data = np.load("youtube_test.npy")

reddit_samples = reddit_data.shape[0]
hackernews_samples = hackernews_data.shape[0]
youtube_samples = youtube_data.shape[0]
samples = reddit_samples + hackernews_samples + youtube_samples

reddit_labels = np.array([0 for i in range(reddit_samples)])
hackernews_labels = np.array([1 for i in range(hackernews_samples)])
youtube_labels = np.array([2 for i in range(youtube_samples)])

data = np.concatenate((reddit_data, hackernews_data, youtube_data), axis=0)
labels = np.concatenate((reddit_labels, hackernews_labels, youtube_labels), axis=0)
data = np.stack((data, labels), axis=1)
np.random.shuffle(data)

x_train = data[0 : samples * 70 // 100, 0]
y_train = data[0 : samples * 70 // 100, 1].astype(int)
class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
class_weights = {0: class_weights[0], 1: class_weights[1], 2: class_weights[2]}
y_train = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_train)))

x_val = data[samples * 70  // 100 : samples * 85 // 100, 0]
y_val = data[samples * 70  // 100 : samples * 85 // 100, 1].astype(int)
y_val = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_val)))

x_test = data[samples * 85 // 100 :, 0]
y_test = data[samples * 85 // 100 :, 1].astype(int)
y_test = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_test)))

tokenizer = Tokenizer(num_words=word_amount)
tokenizer.fit_on_texts(x_train) # må kanskje skrive list(x_train)

f = open("tokenizer.txt", "w")
f.write(str(tokenizer.word_index))
f.close()

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

x_train = pad_sequences(x_train, maxlen=max_comment_length)
x_val = pad_sequences(x_val, maxlen=max_comment_length)
x_test = pad_sequences(x_test, maxlen=max_comment_length)

def evaluate(y_true, y_pred):
	classes = {"[1 0 0]": 0, "[0 1 0]": 1, "[0 0 1]": 2}
	n = len(y_true)
	true = np.zeros(n, dtype=int)
	pred = np.zeros(n, dtype=int)
	for i in range(n):
		true[i] = classes[str(y_true[i])]
		for j in range(3):
			if y_pred[i, j] == np.max(y_pred[i]):
				pred[i] = j
	return classification_report(true, pred, target_names=["Reddit", "Hacker News", "YouTube"])


models = []

model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(3, activation="softmax"))
models.append(model)

model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(LSTM(embedding_dim))
model.add(Dense(3, activation="softmax"))
models.append(model)

model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(Bidirectional(GRU(embedding_dim)))
model.add(Dense(3, activation="softmax"))
models.append(model)

model = Sequential()
model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(GRU(embedding_dim))
model.add(Dense(3, activation="softmax"))
models.append(model)

for i in range(len(models)):
	model = models[i]
	model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
	model.fit(x_train, y_train, class_weight=class_weights, epochs=10, batch_size=128, verbose=1)
	#tfjs.converters.save_keras_model(model, "model")

	f = open("model" + str(i) + ".txt")

	loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
	y_pred_val = model.predict(x_val)
	f.write("Validation: [loss: " + str(loss) + ", accuracy: " + str(accuracy) + "]")
	f.write(evaluate(y_val, y_pred_val))

	loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
	y_pred_val = model.predict(x_val)
	f.write("Test: [loss: " + str(loss) + ", accuracy: " + str(accuracy) + "]")
	f.write(evaluate(y_test, y_pred_test))

	f.close()
