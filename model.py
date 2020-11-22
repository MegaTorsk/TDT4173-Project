import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Seeding for reproducibility
from tensorflow.random import set_seed
random.seed(0)
np.random.seed(0)
set_seed(0)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Attention, Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report
import tensorflowjs as tfjs
import kerastuner as kt

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Optimal hyperparameters, found after optimization

max_comment_length = 100
word_amount = 10000
embedding_dim = 128
lstm_dim = 192
learning_rate = 0.001
epochs = 10


def load_data():
    # Reading data
    # Store as regular python lists instead of numpy arrays
    # Because the data is textstrings, numpy arrays use more space

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
    reddit_data = np.load("formated_data/reddit.npy")
    hackernews_data = np.load("formated_data/hacker_news_test.npy")
    youtube_data = np.load("formated_data/youtube_test.npy")

    reddit_samples = reddit_data.shape[0]
    hackernews_samples = hackernews_data.shape[0]
    youtube_samples = youtube_data.shape[0]
    samples = reddit_samples + hackernews_samples + youtube_samples
    """
    
    # Creating labels
    reddit_labels = np.array([0 for i in range(reddit_samples)])
    hackernews_labels = np.array([1 for i in range(hackernews_samples)])
    youtube_labels = np.array([2 for i in range(youtube_samples)])

    # Concatinating data
    # Numpy arrays has to be concatenated differently
    data = reddit_data + hackernews_data + youtube_data
    #data = np.concatenate((reddit_data, hackernews_data, youtube_data), axis=0)
    labels = np.concatenate((reddit_labels, hackernews_labels, youtube_labels), axis=0)

    # Shuffling data so that the distribution of train and test
    # data are as similar as possible
    data, labels = shuffle(data, labels, random_state=0)

    return data, labels, samples


def export_tokenizer(tokenizer):
    # Write tokenizer in json format
    f = open("tokenizer.json", "w")
    f.write("{")
    i = 0
    for key, value in tokenizer.word_index.items():
        f.write("\"" + str(key) + "\": " + str(value))
        i += 1
        if i == word_amount:
            break
        f.write(", ")
    f.write("}")
    f.close()


def transform_data(data, labels, samples):
    # Splitting the data into train, validation and test.
    x_train = data[0 : samples * 70 // 100]
    x_val = data[samples * 70  // 100 : samples * 85 // 100]
    x_test = data[samples * 85 // 100 : samples]

    y_train = labels[0 : samples * 70 // 100]
    y_val = labels[samples * 70  // 100 : samples * 85 // 100]
    y_test = labels[samples * 85 // 100 : samples]

    # Class weights so that smaller classes are given more weight per sample
    class_weights = class_weight.compute_class_weight("balanced", classes=[0, 1, 2], y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Tokenizer wchich replaces words with tokens
    tokenizer = Tokenizer(num_words=word_amount, oov_token=1)
    tokenizer.fit_on_texts(x_train)

    # Save final tokenizer for use with website
    export_tokenizer(tokenizer)

    # Apply tokenizer to data
    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)
    x_test = tokenizer.texts_to_sequences(x_test)

    # Make sure every data point has the same size
    x_train = pad_sequences(x_train, maxlen=max_comment_length)
    x_val = pad_sequences(x_val, maxlen=max_comment_length)
    x_test = pad_sequences(x_test, maxlen=max_comment_length)

    return (x_train, x_val, x_test), (y_train, y_val, y_test), class_weights


def plot_training(history):
    # Create plots showing accuracy and loss for each epoch

    axis = [i + 1 for i in range(epochs)]
    plt.figure(0)
    # Some versions use "val_acc" instead of "val_accuracy". In case of errors, try the other
    plt.plot(axis, history.history["acc"])
    plt.plot(axis, history.history["val_acc"])
    #plt.plot(axis, history.history["accuracy"])
    #plt.plot(axis, history.history["val_accuracy"])
    plt.ylim(0, 1)
    plt.xticks(axis, axis)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig("model_acc")

    plt.figure(1)
    plt.plot(axis, history.history["loss"])
    plt.plot(axis, history.history["val_loss"])
    plt.ylim(0, max(history.history["loss"] + history.history["val_loss"]) + 0.3)
    plt.xticks(axis, axis)
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig("model_loss")


def export_performance(model, x, y, filename):
    loss, accuracy = model.evaluate(x, y, verbose=1)
    # Convert from one-hot encoding to a list with a values 1, 2 or 3
    y_pred = np.argmax(model.predict(x), axis=1)

    f = open(filename, "w")
    f.write("[loss: " + str(loss) + ", accuracy: " + str(accuracy) + "]\n")
    f.write(classification_report(y, y_pred, target_names=["Reddit", "Hacker News", "YouTube"]))
    f.close()


def train_model(model, x, y, class_weights):
    x_train, x_val, x_test = x
    y_train, y_val, y_test = y

    # Normal categorical crossentropy uses one-hot encoded labels
    # Because we do not, use the sparse version instead (but they are the exact same function)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), class_weight=class_weights, epochs=epochs, batch_size=128, verbose=1)

    plot_training(history)
    export_performance(model, x_val, y_val, "model_val.txt")
    export_performance(model, x_test, y_test, "model_test.txt")

    #model.save("model.hdf5")
    #tfjs.converters.save_keras_model(model, "model")


def create_models():
    models = []

    # LSTM
    model = Sequential()
    model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
    model.add(LSTM(lstm_dim))
    model.add(Dense(3, activation="softmax"))
    models.append(model)

    # Bidirectional LSTM
    model = Sequential()
    model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
    model.add(Bidirectional(LSTM(lstm_dim)))
    model.add(Dense(3, activation="softmax"))
    models.append(model)

    # GRU
    model = Sequential()
    model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
    model.add(GRU(lstm_dim))
    model.add(Dense(3, activation="softmax"))
    models.append(model)

    # Bidirectional GRU
    model = Sequential()
    model.add(Embedding(word_amount, embedding_dim, input_length=max_comment_length))
    model.add(Bidirectional(GRU(lstm_dim)))
    model.add(Dense(3, activation="softmax"))
    models.append(model)

    return models


def hyperparameter_optimization(x, y):
    x_train, x_val, x_test = x
    y_train, y_val, y_test = y

    def model_builder(hp):
        hp_word_amount = hp.Int("word_amount", min_value=5000, max_value=20000, step=5000)
        hp_embedding_dim = hp.Int("embedding_dim", min_value=64, max_value=256, step=64)
        hp_lstm_dim = hp.Int("lstm_dim", min_value=64, max_value=256, step=64)
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

        model = Sequential()
        model.add(Embedding(hp_word_amount, hp_embedding_dim, input_length=max_comment_length))
        model.add(Bidirectional(LSTM(hp_lstm_dim)))
        model.add(Dense(3, activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=hp_learning_rate), metrics=["accuracy"])
        #model.compile(loss="sparse_categorical_crossentropy", optimizer=RMSprop(learning_rate=hp_learning_rate), metrics=["accuracy"])
        return model

    # Some versions use "val_acc" instead of "val_accuracy". In case of errors, try the other
    tuner = kt.Hyperband(model_builder, objective="val_acc", max_epochs=3, factor=3, directory="hyperparam", project_name="comments", seed=0)
    #tuner = kt.Hyperband(model_builder, objective="val_accuracy", max_epochs=3, factor=3, directory="hyperparam", project_name="comments", seed=0)

    tuner.search(x_train, y_train, epochs=3, validation_data=(x_val, y_val), callbacks=[])

    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

    tuner.search_space_summary()
    tuner.results_summary()

    f = open("hyperparam.txt", "w")
    f.write("word_amount: " + str(best_hyperparameters.get("word_amount")) + "\n")
    f.write("embedding_dim: " + str(best_hyperparameters.get("embedding_dim")) + "\n")
    f.write("lstm_dim: " + str(best_hyperparameters.get("lstm_dim")) + "\n")
    f.write("learning_rate: " + str(best_hyperparameters.get("learning_rate")) + "\n")
    f.close()


if __name__ == "__main__":
    data, labels, samples = load_data()
    print("Data loaded")
    x, y, class_weights = transform_data(data, labels, samples)
    print("Data transformed")

    # Hyperparameter optimization
    hyperparameter_optimization(x, y)
    print("Hyperparameter optimization done")

    # Train Bidirectional LSTM
    models = create_models()
    model = models[1]
    train_model(model, x, y, class_weights)
    print("Model finished training")
