#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import sys
import os
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.metrics as tf_metrics
from tensorflow.python.ops.control_flow_ops import cond
import pickle


class Seq2Seq:
    def __init__(self, rnn_cell, rnn_cell_dim, method, vocab_size, logdir, expname, threads=1, seed=42, character_count=50, embedding_matrix=None,
                 max_sentence_length=None, embedding_dim=50, distinct_tags=2):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.contrib.rnn.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.contrib.rnn.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.embedding_matrix = embedding_matrix
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            # self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
            self.targets = tf.placeholder(tf.int64, [None], name='targets')

            self.tags = tf.placeholder(tf.int64, [None])
            self.char_ids = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.charseqs = tf.placeholder(tf.int32, [None, max_sentence_length])
            self.is_first = tf.placeholder(tf.bool)

            self.embedding_placeholder = tf.placeholder_with_default(tf.random_normal([vocab_size, embedding_dim]), [vocab_size, embedding_dim])
            alpha = .3
            print(max_sentence_length)
            embeddings = tf.Variable(initial_value=tf.random_normal([vocab_size, embedding_dim]), dtype=tf.float32)
            encoder_inputs_emb = tf.nn.embedding_lookup(embeddings, ids=self.encoder_inputs)
            shape = tf.convert_to_tensor([-1, max_sentence_length, 50, 1])
            encoder_inputs_emb = tf.reshape(encoder_inputs_emb, shape=shape)

            conv1 = tf_layers.conv2d(encoder_inputs_emb, num_outputs=8, kernel_size=[5, 5])
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

            conv2  = tf_layers.conv2d(pool1, num_outputs=4, kernel_size=[5, 5])
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=3)
            pool2_flat = tf.reshape(pool2, [-1, 7 * 5 * 4])
            dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4)
            self.logits = tf.layers.dense(inputs=dropout, units=2)
            self.labels = tf.one_hot(self.targets, depth=2, dtype=tf.int64)
            self.smax = tf.nn.softmax(self.logits)
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels,
                logits=self.smax,
            )


            # mask = tf.sequence_mask(self.sentence_lens, tf.reduce_max(self.sentence_lens))
            # logits = alpha * outputs[0] + (1 - alpha) * outputs[1]
            self.loss = tf.reduce_mean(stepwise_cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.training = self.optimizer.minimize(self.loss)
            self.predictions = tf.argmax(self.logits, 1)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.targets)

            self.dataset_name = tf.placeholder(tf.string, [])

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, encoder_in, targets, first=False):
        feed_dict = {self.encoder_inputs: encoder_in,
                     self.targets: targets,
                     self.dataset_name: "train", self.is_first: first}
        _, loss, acc  = self.session.run([self.training, self.loss, self.accuracy], feed_dict)
        return loss, acc

    def train_with_chars(self, sentence_lens, charseqs_ids, tags, charseqs, charseq_lens):

        feed_dict = {self.sentence_lens: sentence_lens, self.char_ids: charseqs_ids, self.charseqs: charseqs,
                     self.tags: tags, self.dataset_name: "train", self.charseq_lens: charseq_lens}
        _, summary = self.session.run([self.training], feed_dict)

    def evaluate_with_chars(self, sentence_lens, charseqs_ids, tags, charseqs, charseq_lens):

        feed_dict = {self.sentence_lens: sentence_lens, self.char_ids: charseqs_ids, self.charseqs: charseqs,
                     self.tags: tags, self.dataset_name: "dev", self.charseq_lens: charseq_lens}
        accuracy, summary = self.session.run([self.accuracy], feed_dict)

        return accuracy

    def predict_with_chars(self, sentence_lens, charseqs_ids, tags, charseqs, charseq_lens):

        feed_dict = {self.sentence_lens: sentence_lens, self.char_ids: charseqs_ids, self.charseqs: charseqs,
                     self.tags: tags, self.dataset_name: "test", self.charseq_lens: charseq_lens}
        predictions = self.session.run(self.predictions, feed_dict)
        return predictions

    def evaluate(self, sentence_lens, forms, tags):
        feed_dict = {self.sentence_lens: sentence_lens, self.forms: forms,
                     self.tags: tags, self.dataset_name: "dev", self.is_first: False}
        accuracy = self.session.run([self.accuracy], feed_dict)
        return accuracy

    def predict(self, inputs):

        feed_dict = {self.encoder_inputs: inputs, self.is_first: False}
        return self.session.run(self.predictions, feed_dict)

    def save(self, model_name='models/model_final'):
        saver = tf.train.Saver()
        saver.save(self.session, model_name)

def tr_embeddings(word_dataset, embedding_map, word_ids, offset):
    new_word_ids = np.ones(word_ids.shape, dtype=np.int32)
    word_map = word_dataset.factors[0]['words']
    for i, ax in enumerate(word_ids):
        for j, wi in enumerate(ax):
            try:
                new_word_ids[i, j] = embedding_map[word_map[word_ids[i,j]]]
            except:
                embedding_map[word_map[word_ids[i,j]]] = offset
                new_word_ids[i, j] = offset
                offset += 1

    return new_word_ids, offset


def get_oob_words(dataset, embedding_map, word_ids):
    words = set()
    for i, ax in enumerate(word_ids):
        for j, wi in enumerate(ax):
            try:
                embedding_map[dataset.factors[dataset.FORMS]['words'][word_ids[i,j]]]
            except:
                words.add(word_ids[i,j])

    return words


def pad_charseqs(charseqs, max):
    result = np.zeros((charseqs.shape[0], max))
    for i, ax in enumerate(charseqs):
        for j, x in enumerate(ax):
            result[i,j] = x
    return result


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="en-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="en-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="en-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
    # used_method = "learned_we"
    # used_method = "updated_pretrained_we"
    used_method = "only_pretrained_we"
    # used_method = "char_rnn"
    # used_method = "char_conv"
    # used_method = "charagram"
    parser.add_argument("--method", default=used_method, type=str, help="Which method of word embeddings to use.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--exp", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "tagger-{}{}-m{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.method, args.batch_size, args.epochs)

    from dataset import Dataset

    dataset = Dataset(fn='data/derismall.tsv', as_chars=True)
    test_data = dataset.get_test()
    train_data = dataset.get_train()
    network = Seq2Seq(rnn_cell="GRU", rnn_cell_dim=20, logdir=args.logdir, method="learned_we",
                      expname=args.exp, threads=args.threads, vocab_size=len(dataset.chars2ints) + 2)
    with open("test.reference", 'wb') as f:
        pickle.dump(test_data[1], f)
    # Train
    for i in range(args.epochs):
        while dataset.has_next():
            features, labels, sentence_lens = dataset.next_batch()
            network.train(sentence_lens, features, labels)
        dataset.reset()
        # network.evaluate("dev", mnist.validation.images, mnist.validation.labels)
        eval_res = network.evaluate(test_data[2], test_data[0], test_data[1])
        # network.predict(str(i) + "test", test_data[0], test_data[1])
        # network.predict(str(i) + "train", train_data[0], train_data[1])
        print(eval_res)
