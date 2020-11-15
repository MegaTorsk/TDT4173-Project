import random
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.random import set_seed
random.seed(0)
np.random.seed(0)
set_seed(0)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Attention, Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import classification_report
import tensorflowjs as tfjs

model = load_model("model.hdf5")
tfjs.converters.save_keras_model(model, "model")