from __future__ import division
from __future__ import print_function

import pickle
import datetime
import sys  

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

NUMBER_FEATURES = 107
HIDDEN = 200
LABELS = 2

def evaluate(gold, hyp):
    tp = fp = tn = fn = 0
    for g, h in zip(gold, hyp):
        if g:
            if h == 1:
                tp += 1
            else:
                fn += 1
        else:
            if h == 0:
                tn += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)
    return precision, recall, fscore


class Network:
    def __init__(self, logdir, experiment, threads, layer_size):
        # Construct the graph
        with tf.name_scope("inputs"):
            self.input = tf.placeholder(tf.float32, [None, NUMBER_FEATURES], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

        hidden_layer1 = layers.fully_connected(self.input, num_outputs=layer_size, activation_fn=tf.nn.sigmoid,
                                              scope="hidden_layer1")
        #hidden_layer2 = layers.fully_connected(hidden_layer1, num_outputs=int(layer_size * 0.5), activation_fn=tf.nn.sigmoid,
         #                                     scope="hidden_layer2")
        output_layer = layers.fully_connected(hidden_layer1, num_outputs=LABELS, activation_fn=None,
                                              scope="output_layer")

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=self.labels), name="loss")
            tf.summary.scalar("training/loss", loss)
        with tf.name_scope("train"):
            self.training = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope("accuracy"):
            predictions = tf.argmax(output_layer, 1, name="predictions")
            self.predictions = predictions
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, self.labels), tf.float32), name="accuracy")
            self.accuracy = accuracy
            tf.summary.scalar("training/accuracy", accuracy)

        # Summaries
        self.summaries = {'training': tf.summary.merge_all() }
        for dataset in ["valid", "test", "train"]:
            self.summaries[dataset] = tf.summary.scalar(dataset + "/accuracy", accuracy)

        # Create the session
        self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                        intra_op_parallelism_threads=threads))

        self.session.run(tf.initialize_all_variables())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.summary.FileWriter("{}/{}-{}".format(logdir, timestamp, experiment), graph=self.session.graph, flush_secs=10)
        self.steps = 0

    def train(self, features, labels):
        self.steps += 1
        _, summary = self.session.run([self.training, self.summaries['training']], {self.input: features, self.labels: labels})
        self.summary_writer.add_summary(summary, self.steps)

    def evaluate(self, epoch, dataset, features, labels):
        summary = self.summaries[dataset].eval({self.input: features, self.labels: labels}, self.session)
        self.summary_writer.add_summary(summary, self.steps)
        _, acc = self.session.run([self.training, self.accuracy], {self.input: features, self.labels: labels})
        with open("one_layer_epoch_{}.acc".format(epoch), 'w') as f:
            f.write(str(acc))

    def predict(self, epoch, features, labels):
        _, pred = self.session.run([self.training, self.predictions], {self.input: features, self.labels: labels})
        precision, recall, fscore = evaluate(labels, pred)
        with open("one_layer_epoch_{}.scores".format(epoch), 'w') as f:
            f.write("{},{},{}".format(precision, recall, fscore))



if __name__ == '__main__':
    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=50, type=int, help='Batch size.')
    parser.add_argument('--layer_size', default=50, type=int, help='Batch size.')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
    parser.add_argument('--logdir', default="logs", type=str, help='Logdir name.')
    parser.add_argument('--exp', default="2-mnist-annotated-graph", type=str, help='Experiment name.')
    parser.add_argument('--threads', default=1, type=int, help='Maximum number of threads to use.')
    args = parser.parse_args()

    # Load the data

    # Construct the network
    network = Network(logdir=args.logdir, experiment=args.exp, threads=args.threads, layer_size=args.layer_size)
    from dataset import Dataset
    dataset = Dataset(fn='data/derinet-1.4-shuffled.tsv')
    test_data = dataset.get_test()
    train_data = dataset.get_train()
    valid_data = dataset.get_valid()
#    with open("test.reference", 'w') as f:
#        for p in test_data[1]:
#            f.write(str(p))
#    with open("train.reference", 'w') as f:
#        for p in train_data[1]:
#            f.write(str(p))
#    with open("valid.reference", 'w') as f:
#        for p in valid_data[1]:
#           f.write(str(p))
#    # Train
    for i in range(args.epochs):
        while dataset.has_next():
            features, labels = dataset.next_batch()
            network.train(features, labels)
        dataset.reset()

        network.evaluate(args.exp + str(i) + "_test", "test", test_data[0], test_data[1])
        network.evaluate(args.exp + str(i) + "_valid", "valid", valid_data[0], valid_data[1])
        network.evaluate(args.exp + str(i) + "_test", "train", train_data[0], train_data[1])
        network.predict(args.exp + str(i) + "_test", test_data[0], test_data[1])
        network.predict(args.exp + str(i) + "_train", train_data[0], train_data[1])
        network.predict(args.exp + str(i) + "_valid", valid_data[0], valid_data[1])
