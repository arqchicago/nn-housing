# -*- coding: utf-8 -*-
"""
@author: Ahmad Qadri
Sequential Neural Network with dropout hidden layer and softmax output layer on Housing Dataset

"""
import pandas as pd
import numpy as np
import sklearn.model_selection as skms
from time import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras import utils


seed = 5941
tf.random.set_seed(seed)

#----  features and target variable
quant_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'age', 'appliance_age', 'crime', 'renovated'] 
cat_features = ['backyard', 'view', 'condition']    
features = quant_features

target_var_orig = 'price'
target_var = 'high_priced'
weight = 'weight'
sqft = 'sqft_living'

#----  uploading data
housing_df = pd.read_csv('data\\housing.csv')
housing_df[target_var] = 0
housing_df.loc[housing_df[target_var_orig]>350000, target_var] = 1
housing_df.loc[housing_df[target_var_orig]>550000, target_var] = 2
rows, cols = housing_df.shape
print(f'> rows = {rows},  cols = {cols}')

#----  train/test split
X, y = housing_df[features], housing_df[target_var]
X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.20, random_state = seed)

y_train_dummy = utils.to_categorical(y_train)
y_test_dummy = utils.to_categorical(y_test)

X_train_rows, y_train_rows = X_train.shape[0], y_train.shape[0]
X_test_rows, y_test_rows = X_test.shape[0], y_test.shape[0] 

#X_train_weights = housing_df[weight].loc[X_train.index.values]
#X_test_weights = housing_df[weight].loc[X_test.index.values]

X_train_hp_rows = housing_df.groupby(target_var).size().to_frame('size').reset_index().values.tolist()
X_test_hp_rows = housing_df.groupby(target_var).size().to_frame('size').reset_index().values.tolist()

print(f'> features = {len(features)}')
print(f'> training set = {X_train_rows} ({round(X_train_rows*1.0/rows,3)})')
print(f'> testing set = {X_test_rows} ({round(X_test_rows*1.0/rows,3)})\n')
print(f'> training set price dummy = {X_train_hp_rows}')
print(f'> testing set price dummy = {X_test_hp_rows}\n')

#----  creating the model
model = keras.Sequential()
model.add(keras.Input(shape=(9,)))
model.add(Dense(18, activation='sigmoid', name='layer1'))
model.add(Dense(36, activation='sigmoid', name='layer2'))
model.add(Dropout(0.10, name='dropout'))
model.add(Dense(18, activation='sigmoid', name='layer3'))
model.add(Dense(3, activation='softmax', name='output_layer'))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

print(model.summary())
print(f'input shape= {model.input_shape}')
print(f'output shape= {model.output_shape}')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_dummy, epochs=500, validation_split=0.10, verbose=1, callbacks=[tensorboard])

test_set_loss, test_set_accuracy = model.evaluate(X_test, y_test_dummy)
print(f'test set loss = {round(test_set_loss, 4)}  test set accuracy = {round(test_set_accuracy, 4)}')

