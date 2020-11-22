import numpy as np
import pickle
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#import tensorflowjs as tfjs


# Seeding for reproducibility
np.random.seed(0)

# Update to the directory where you saved the
# preprocessed data
dataset_directory = "formated_data/"

# Reading data
reddit_file0 = open(dataset_directory + "reddit0.pickle", "rb")
reddit_data0 = pickle.load(reddit_file0)
reddit_file0.close()
reddit_file1 = open(dataset_directory + "reddit1.pickle", "rb")
reddit_data1 = pickle.load(reddit_file1)
reddit_file1.close()
reddit_file2 = open(dataset_directory + "reddit2.pickle", "rb")
reddit_data2 = pickle.load(reddit_file2)
reddit_file2.close()
reddit_file3 = open(dataset_directory + "reddit3.pickle", "rb")
reddit_data3 = pickle.load(reddit_file3)
reddit_file3.close()

reddit_data = reddit_data0 + reddit_data1 + reddit_data2 + reddit_data3 # Combine lists

hackernews_file = open(dataset_directory + "hacker_news.pickle", "rb")
hackernews_data = pickle.load(hackernews_file)
hackernews_file.close()

youtube_file = open(dataset_directory + "youtube.pickle", "rb")
youtube_data = pickle.load(youtube_file)
youtube_file.close()

reddit_samples = len(reddit_data)
hackernews_samples = len(hackernews_data)
youtube_samples = len(youtube_data)
samples = reddit_samples + hackernews_samples + youtube_samples


# Creating labels
reddit_labels = [0 for i in range(reddit_samples)]
hackernews_labels = [1 for i in range(hackernews_samples)]
youtube_labels = [2 for i in range(youtube_samples)]

# Concatinating data
data = reddit_data + hackernews_data + youtube_data
labels = reddit_labels + hackernews_labels + youtube_labels

# Shuffling data so that the distribution of train and test
# data are as similar as possible
data, labels = shuffle(data, labels, random_state=0)

# Splitting the data into train, validation and test.
x_train = data[0 : samples * 70 // 100]
x_val = data[samples * 70  // 100 : samples * 85 // 100]
x_test = data[samples * 85 // 100 :]

y_train = labels[0 : samples * 70 // 100]
y_val = labels[samples * 70  // 100 : samples * 85 // 100]
y_test = labels[samples * 85 // 100 :]

# Class weights so that smaller classes are given more weight per sample
class_weights = class_weight.compute_class_weight("balanced", [0, 1, 2], y=y_train)
class_weights = dict(enumerate(class_weights))

# Vectorize using the number of words of each type
bow_converter = CountVectorizer(tokenizer=lambda doc: doc)

x_train = bow_converter.fit_transform(x_train)
x_val = bow_converter.transform(x_val)
x_test = bow_converter.transform(x_test)


# Create model
model = LogisticRegression(class_weight=class_weights, max_iter=10000).fit(x_train, y_train)

# Print results
val_score = model.score(x_val, y_val)
print("Validation: [accuracy: " + str(val_score) + "]")
test_score = model.score(x_test, y_test)
print("Test: [accuracy: " + str(test_score) + "]")