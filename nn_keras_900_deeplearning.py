#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Importing standard libraries


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


## Importing sklearn libraries

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



## Keras Libraries for Neural Networks

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD


## Read data from the CSV file

train = pd.read_csv('train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)



## We will be working with categorical crossentropy function
## It is required to further convert the labels into "one-hot" representation

y_cat = to_categorical(y_train)
print(y_cat.shape)

## hyperparameters
filename = '150_ep_drop_0.7'
learning_rate = 0.01
momentum = 0.8
decay = 0.0
batch_size = 128
epochs = 150
dropout = 0.7

print("Hyperparameters:")
print("="*25)
print("learning_rate: ", learning_rate)
print("momentum: ", momentum)
print("decay: ", decay)
print("batch size: ", batch_size)
print("no. epochs: ", epochs)
print("dropout: ", dropout)
print("-"*25)
print()


## Developing a layered model for Neural Networks
## Input dimensions should be equal to the number of features
## We used softmax layer to predict a uniform probabilistic distribution of outcomes

model = Sequential()
model.add(Dense(2048,input_dim=192,  init='uniform', activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(99, activation='softmax'))

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay, nesterov=False)## Error is measured as categorical crossentropy or multiclass logloss
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics = ["accuracy"])

summary = str(model.summary())
print(summary)
print()

## Fitting the model on the whole training data
history = model.fit(x_train,y_cat,batch_size=batch_size,
                    nb_epoch=epochs,verbose=1, validation_split=0.1)


## Writing file with report

out = open(filename + 'report.txt','w')
out.write('Hyperparameters')
out.write('\n')
out.write('='*25)
out.write('\n')
out.write('\n')
out.write("learning_rate: {0}".format(learning_rate))
out.write('\n')
out.write("momentum: {0}".format(momentum))
out.write('\n')
out.write("decay: {0}".format(decay))
out.write('\n')
out.write("batch size: {0}".format(batch_size))
out.write('\n')
out.write("no. epochs: {0}".format(epochs))
out.write('\n')
out.write("dropout: {0}".format(dropout))
out.write('\n')
out.write("-"*25)
out.write('\n')
out.write('\n')
out.write(summary)
out.write('\n')
out.write('\n')
out.write('val_acc: {0}'.format(max(history.history['val_acc'])))
out.write('\n')
out.write('val_loss: {0}'.format(min(history.history['val_loss'])))
out.write('\n')
out.write('train_acc: {0}'.format(max(history.history['acc'])))
out.write('\n')
out.write('train_loss: {0}'.format(min(history.history['loss'])))
out.write('\n')
out.write("train/val loss ratio: {0}".format(min(history.history['loss'])/min(history.history['val_loss'])))
out.close()




## we need to consider the loss for final submission to leaderboard
## print(history.history.keys())
print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))

print()
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))

# summarize history for loss
## Plotting the loss with the number of iterations

plt.figure()
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(filename + "_loss"+".png")
#plt.show()

## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(filename + "_error" + ".png")
#plt.show()


## read test file
test = pd.read_csv('test.csv')
index = test.pop('id')
test = StandardScaler().fit(test).transform(test)
yPred = model.predict_proba(test)

## Converting the test predictions in a dataframe as depicted by sample submission

submission = pd.DataFrame(yPred, index=index, columns=le.classes_)
submission.to_csv(filename + '.csv')