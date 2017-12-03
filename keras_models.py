import numpy as np
import logging

from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Embedding
from keras.utils import np_utils
from keras.backend import argmax



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
                      optimizer='rmsprop',metrics=['accuracy'])
        self.model = model
        self.layer_count = len(layer_sizes)
        self.activation = activation

    def _process_data(self, X, y):
        y = np_utils.to_categorical(y, num_classes=2)
        return X, y

    def train(self, X, y, epochs=20, batch_size=128):
        X, y = self._process_data(X, y)
        return self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)

    def evaluate(self, X, y):
        X, y = self._process_data(X, y)
        score = self.model.evaluate(X, y, batch_size=128, verbose=1)
        return score

    def predict(self, X):
        predicted = self.model.predict(X)
        predicted = np.argmax(predicted, axis=1)
        return predicted

class Seq2Seq:

    def __init__(self, latent_dim=40, num_tokens=0, max_len=0):
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None,))
        embeddings = Embedding(num_tokens, 200, input_length=max_len, mask_zero=True)
        self.encoder_inputs_emb = embeddings(self.encoder_inputs)
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs_emb)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None,))
        self.decoder_inputs_emb = embeddings(self.decoder_inputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs_emb,
                                             initial_state=self.encoder_states)
        self.decoder_dense = Dense(num_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(decoder_outputs)
        # self.decoder_outputs = Lambda(lambda x: argmax(x, axis=1), dtype='float64')(self.decoder_dense(decoder_outputs))

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        self.model = model
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens

        # self.model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
        # self.model.compile(loss='mse', optimizer='rmsprop')

    def train(self, X, y, epochs=20, batch_size=128, new_dim = 0,dataset=None):
        # y_shifted = np.zeros((y.shape[0], y.shape[1], y.shape[2]))
        # for i, example in enumerate(y):
        #     empty_line = np.zeros((1, self.num_tokens))
        #     empty_line[0, 2] = 1
        #     shifted_example = np.concatenate((empty_line, example[:-1]))
        #     y_shifted[i, :, :] = shifted_example

        y_shifted = np.zeros((y.shape[0], y.shape[1]))
        for i, example in enumerate(y):
            shifted_example = np.concatenate((np.array([2]), example[:-1]))
            y_shifted[i, :] = shifted_example
        self.model.fit([X, y_shifted], dataset.transform2onehot(y, new_dim), batch_size=batch_size, epochs=epochs, verbose=1, validation_split=.2)

    def evaluate(self, X, y):
        X, y = self._process_data(X, y)
        score = self.model.evaluate(X, y, batch_size=128, verbose=1)
        return score

    def predict(self, X, chars2ints, ints2chars):
        print(X)
        self.prediction_encoder_model = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.prediction_decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        return self._decode_sequence(X, chars2ints, ints2chars)

    def _decode_sequence(self, input_seq, chars2ints, ints2chars):
        # Encode the input as state vectors.
        input_seq = np.array(input_seq)
        states_value = self.prediction_encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, self.num_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, chars2ints['&']] = 1

        # Sampling loop for a batch of sequences
        stop_condition = False
        decoded_string = []
        while not stop_condition:
            output_tokens, h, c = self.prediction_decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_string.append(ints2chars[sampled_token_index])

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_token_index == chars2ints['#'] or
                        len(decoded_string) > 200):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_tokens))
            target_seq[0, 0, sampled_token_index] = 1

            # Update states
            states_value = [h, c]

        return decoded_string

class Seq2Classify:

    def __init__(self, latent_dim=40, num_tokens=0, max_len=0):
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None,))
        embeddings = Embedding(num_tokens, 200, input_length=max_len, mask_zero=True)
        self.encoder_inputs_emb = embeddings(self.encoder_inputs)
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs_emb)
        softmax = Dense(2, activation='softmax')
        self.outputs = softmax(encoder_outputs)
        model = Model([self.encoder_inputs], self.outputs)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['accuracy'])
        self.model = model
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens

        # self.model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
        # self.model.compile(loss='mse', optimizer='rmsprop')

    def train(self, X, y, epochs=20, batch_size=128, new_dim = 0,dataset=None):
        # y_shifted = np.zeros((y.shape[0], y.shape[1], y.shape[2]))
        # for i, example in enumerate(y):
        #     empty_line = np.zeros((1, self.num_tokens))
        #     empty_line[0, 2] = 1
        #     shifted_example = np.concatenate((empty_line, example[:-1]))
        #     y_shifted[i, :, :] = shifted_example
        self.model.fit([X], np_utils.to_categorical(y, num_classes=2), batch_size=batch_size, epochs=epochs, verbose=1, validation_split=.2)

    def predict(self, X):
        predicted = self.model.predict(X)
        predicted = np.argmax(predicted, axis=1)
        return predicted
