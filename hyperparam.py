import numpy as np
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Attention, Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report
import kerastuner as kt
#import tensorflowjs as tfjs

np.random.seed(0)

import tensorflow.compat.v1 as tfcompat

tfcompat.disable_v2_behavior()  #disable for tensorFlow V2
physical_devices = tfcompat.config.experimental.list_physical_devices("GPU")
tfcompat.config.experimental.set_memory_growth(physical_devices[0], True)


max_comment_length = 100
word_amount = 10000
embedding_dim = 128
lstm_units = 128


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

"""
reddit_data = np.load("reddit.npy")
hackernews_data = np.load("hacker_news_test.npy")
youtube_data = np.load("youtube_test.npy")

reddit_samples = reddit_data.shape[0]
hackernews_samples = hackernews_data.shape[0]
youtube_samples = youtube_data.shape[0]
samples = reddit_samples + hackernews_samples + youtube_samples
"""

reddit_labels = np.array([0 for i in range(reddit_samples)])
hackernews_labels = np.array([1 for i in range(hackernews_samples)])
youtube_labels = np.array([2 for i in range(youtube_samples)])

#data = np.concatenate((reddit_data, hackernews_data, youtube_data), axis=0)
data = reddit_data + hackernews_data + youtube_data
labels = np.concatenate((reddit_labels, hackernews_labels, youtube_labels), axis=0)
data, labels = shuffle(data, labels, random_state=0)

samples = samples // 100

x_train = data[0 : samples * 70 // 100]
x_val = data[samples * 70  // 100 : samples * 85 // 100]
x_test = data[samples * 85 // 100 : samples]

y_train = labels[0 : samples * 70 // 100]
y_val = labels[samples * 70  // 100 : samples * 85 // 100]
y_test = labels[samples * 85 // 100 : samples]

class_weights = class_weight.compute_class_weight("balanced", [0, 1, 2], y=y_train)
class_weights = dict(enumerate(class_weights))

tokenizer = Tokenizer(num_words=word_amount, oov_token=0)
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

#y_train = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_train)))
#y_val = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_val)))
#y_test = np.array(list(map(lambda x: [int(i == x) for i in range(3)], y_test)))

"""
models = []

# Logistic regression
model = Sequential()
#model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
model.add(Dense(3, activation="softmax"))
#models.append(model)

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

for i in range(len(models)):
	model = models[i]
	model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
	model.fit(x_train, y_train, class_weight=class_weights, epochs=3, batch_size=128, verbose=1)
	#tfjs.converters.save_keras_model(model, "model" + str(i))

	f = open("model" + str(i) + ".txt", "w")

	loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
	f.write("Validation: [loss: " + str(loss) + ", accuracy: " + str(accuracy) + "]")
	y_val_pred = model.predict(x_val)
	f.write(classification_report(np.argmax(y_val, axis=1), np.argmax(y_val_pred, axis=1), target_names=["Reddit", "Hacker News", "YouTube"]))

	loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
	f.write("Test: [loss: " + str(loss) + ", accuracy: " + str(accuracy) + "]")
	y_test_pred = model.predict(x_test)
	f.write(classification_report(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1), target_names=["Reddit", "Hacker News", "YouTube"]))

	f.close()
"""

# Hyperparameter tuning with keras-tuner

def model_builder(hp):
	#hp_word_amount = hp.Int("word_amount", min_value=5000, max_value=20000, step=5000)
	#hp_embedding_dim = hp.Int("embedding_dim", min_value=64, max_value=256, step=64)
	#hp_max_comment_length = hp.Int("max_comment_length", min_value=50, max_value=300, step=50)
	#hp_lstm_units = hp.Int("lstm_units", min_value=64, max_value=256, step=64)
	#hp_optimizer = hp.Choice("optimizer", values=[Adam(learning_rate=1e-2), Adam(learning_rate=1e-3), Adam(learning_rate=1e-4), RMSprop(learning_rate=1e-2), RMSprop(learning_rate=1e-3), RMSprop(learning_rate=1e-4)])
	hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
	#hp_optimizer = hp.Choice("optimizer", ["adam", "sgd", "rmsprop"])
	model = Sequential()
	model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
	model.add(Bidirectional(LSTM(lstm_units)))
	model.add(Dense(3, activation="softmax"))
	model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=hp_learning_rate), metrics=["accuracy"])
	return model

tuner = kt.Hyperband(model_builder, objective="val_acc", max_epochs=3, factor=3, directory="hyperparam", project_name="comments", seed=0)

tuner.search(x_train, y_train, epochs=3, validation_data=(x_val, y_val), callbacks=[])

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

tuner.search_space_summary()
tuner.results_summary()

#f = open("hyperparam.txt", "w")
#f.write("word_amount: " + str(best_hyperparameters.get("word_amount")) + "\n")
#f.write("embedding_dim: " + str(best_hyperparameters.get("embedding_dim")) + "\n")
#f.write("max_comment_length: " + str(best_hyperparameters.get("max_comment_length")) + "\n")
#f.write("lstm_units: " + str(best_hyperparameters.get("lstm_units")) + "\n")
#f.write("learning_rate: " + str(best_hyperparameters.get("learning_rate")) + "\n")
#f.write("optimizer: " + str(best_hyperparameters.get("optimizer")) + "\n")
#f.close()

print(best_hyperparameters.get("learning_rate"))