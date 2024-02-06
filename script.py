import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# Load & Inspect Data
dataset = pd.read_csv('admissions_data.csv')
print(dataset.describe)

# Define Labels and Features
dataset.drop(['Serial No.'], axis=1)
labels = dataset.iloc[:,-1]
features = dataset.iloc[:, 0:-1]

# Split Data into Training Set and Test Sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardise Numerical Features
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only_numeric", StandardScaler(), numerical_columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.fit_transform(features_test)

# Create Neural Network
model = Sequential()
input = InputLayer(input_shape = (features.shape[1], ))
model.add(input)
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
print(model.summary())

# Initilising the optimizer and compiling the model
opt = Adam(learning_rate=0.01)
model.compile(loss="mse", metrics=["mae"], optimizer=opt)

# Fit and evaluate the model
history = model.fit(features_train_scaled, labels_train, epochs=10, batch_size=1, verbose=1, validation_data=(features_test_scaled, labels_test))
res_mse, res_mae = model.evaluate(features_test_scaled, labels_test, verbose=0)
print(res_mse, res_mae)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae']) 
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss']) 
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

fig.tight_layout()
fig.savefig('images/model_plots.png')