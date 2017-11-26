import numpy as np
import logging

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
#
# Y_train = np_utils.to_categorical(y_train, 10)
# Y_test = np_utils.to_categorical(y_test, 10)

class FeedForward:

    def __init__(self, layer_sizes=None, input_dim=107, activation='relu'):
        model = Sequential()
        if layer_sizes is None:
            layer_sizes = [10]
        model.add(Dense(output_dim=layer_sizes[0], activation=activation, input_dim=input_dim))
        if len(layer_sizes) > 1:
            for layer_size in layer_sizes[1:]:
                model.add(Dense(output_dim=layer_size, activation=activation))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',metrics=['accuracy'])
        self.model = model
        self.layer_count = len(layer_sizes)
        self.activation = activation

    def _process_data(self, X, y):
        y = np_utils.to_categorical(y, nb_classes=2)
        return X, y

    def train(self, X, y, epochs=20, batch_size=128):
        X, y = self._process_data(X, y)
        self.model.fit(X, y, batch_size=batch_size, nb_epoch=epochs, verbose=1)

    def evaluate(self, X, y):
        X, y = self._process_data(X, y)
        score = self.model.evaluate(X, y, batch_size=128, verbose=1)
        return score

    def predict(self, X):
        predicted = self.model.predict(X)
        predicted = np.argmax(predicted, axis=1)
        return predicted
