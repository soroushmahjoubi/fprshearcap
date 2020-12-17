# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:16:38 2020
I used the following tutorial to build this code:
https://www.tensorflow.org/tutorials/keras/regression

@author: Soroush Mahjoubi
"""

# Import the platforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn import metrics

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)



# The input variables and the target (Vutot) which is the shear capacity
column_names = ['bw',	'h'	,'d'	,'lspan'	,'r'	,'a/d'	,'av/d'	,'da',	'fc,cyl'	,'Fiber Type'	,'Vf',	'lf/df',	'ftenf',	'Vutot']

# read  the dataset
raw_dataset = pd.read_csv("Dataset.csv", names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)

dataset = raw_dataset.copy()

# Split the data into train and test
train_dataset = dataset.sample(frac=0.7, random_state=8)
test_dataset = dataset.drop(train_dataset.index)

# show the overall statistics
train_dataset.describe().transpose()

# Seperate the target variable from the input variables
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Vutot')
test_labels = test_features.pop('Vutot')

# Normalize the data
normalizer = preprocessing.Normalization()


def build_compile_NN_model(norm):
# Generate a simple Neural network with three hidden layers that have 64 nodes and relu nonlinearity as their activation function
  model = keras.Sequential([
      norm,
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(1,activation='relu')
  ])
  
  # Compile the model and consider mean absolute error and adam as the loss function and the optimizer
  model.compile(loss='mse',
              optimizer='adam')
  return model


dnn_model = build_compile_NN_model(normalizer)

# Fit the model to the dataset
# consider 20% of the data for validating the training process
# 200 number of epochs to train the model
history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=200)

# Generate a function to plot the learning curves for the training and validation data
def plot_loss(history):
  plt.plot(history.history['loss'], label='Training loss',color='green')
  plt.plot(history.history['val_loss'], label='Validation loss',color= 'red')
  
  # Make a lambda function to determine the min and max of the vertical axis
  truncate = lambda n: round(round((max(n)*50)/50))
  plt.ylim([0,truncate(history.history['loss'])])

  plt.xlabel('Epoch')
  plt.ylabel('Mean absolute error [KN]')
  plt.legend()
  

# Plot the loss function during the training process
plot_loss(history)
plt.savefig('learning_curves.png')

# Plot actual target outputs versus the predicted values
a = plt.axes(aspect='equal')


# Determine the the coefficient of determination (R2)
trainr2=metrics.r2_score(train_labels, dnn_model.predict(train_features))
testr2=metrics.r2_score(test_labels, dnn_model.predict(test_features))

string_in_string = "Train Set R2 = {:0.2f}".format(trainr2)
plt.scatter(train_labels, dnn_model.predict(train_features),color= 'orange',label=string_in_string)

string_in_string = "Test Set R2 = {:0.2f}".format(testr2)
plt.scatter(test_labels, dnn_model.predict(test_features),color='green',label=string_in_string)
plt.xlim([0,1200])
plt.ylim([0,1200])

# Diagonal line (zero error)
_ = plt.plot([0,1200], [0,1200],ls='--',lw=1.5,alpha=1,color= '#A2A2A1FF',label='Zero error')

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()

plt.savefig('predicted_actual.png')

