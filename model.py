from __future__ import division
from __future__ import print_function

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from keras_models import FeedForward, Seq2Seq, Seq2Classify
from utils import evaluate, measure_accuracy
from model_rnn import Seq2Seq

if __name__ == '__main__':

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size.')
    parser.add_argument('--layer_sizes', default='50,15', type=str, help='Sizes of respective layers; comma separated')
    parser.add_argument('--cell_dim', default=50, type=int)
    parser.add_argument('--cell_type', default='GRU')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.')
    parser.add_argument('--logdir', default="logs", type=str, help='Logdir name.')
    parser.add_argument('--exp', default="2-mnist-annotated-graph", type=str, help='Experiment name.')
    parser.add_argument('--threads', default=1, type=int, help='Maximum number of threads to use.')
    args = parser.parse_args()

    # Load the data

    # Construct the network
    from dataset import Dataset
    dataset = Dataset(fn='data/derismall.tsv', as_chars=True)
    test_X, test_y, _ = dataset.get_test()
    train_X, train_y, _ = dataset.get_train()
    # valid_data = dataset.get_valid()
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
    layer_sizes = [int(ls) for ls in args.layer_sizes.split(',')]
    # model = FeedForward(layer_sizes=layer_sizes, activation=args.activation)
    # model = Seq2Classify(latent_dim=5, num_tokens=dataset.number_tokens, max_len=dataset.max_token_length)
    model = Seq2Seq(args.cell_type, args.cell_dim, method='', vocab_size=dataset.vocab_size, logdir='.', expname=args.exp)
    eos = dataset.chars2ints['EOS']
    for epoch in range(args.epochs):
        print('Training epoch', epoch)
        i = 0
        dataset.reset()
        while dataset.has_next():
            i += 1
            inputs, targets = dataset.next_batch()
            loss = model.train(encoder_in=inputs, decoder_in=np.ones((1, dataset.max_token_length)), decoder_targets=targets)
            if i % 100 == 0:
                print(loss)
            #     prediction = model.predict(inputs)
            #     inputs_mapped = ''.join([dataset.int2chars[ch] for ch in inputs[0] if ch != 0])
            #     prediction_mapped = ''.join([dataset.int2chars[ch] for ch in prediction[0] if ch != 0])
            #
            #     print('In: {}\nPredict: {}\n\n'.format(inputs_mapped, prediction_mapped))

    os.makedirs('models', exist_ok=True)
    model.save('models/{}'.format(args.exp))
    # hist = model.train(train_X, train_y, epochs=args.epochs)
    # with open('model.bin', 'wb') as f:
    #     pickle.dump(model, f)

    # with open('model.bin', 'rb') as f:
    #     model = pickle.load(f)
    # evaluated = model.evaluate(test_X, test_y)
    predicted = model.predict(test_X)
    def map2chars(seq):
        chars = ''.join([dataset.int2chars[ch] for ch in seq if ch != 0])
        return chars

    inputs_mapped = [map2chars(tx) for tx in test_X]
    prediction_mapped = [map2chars(pred) for pred in predicted]

    os.makedirs(args.exp, exist_ok=True)
    with open(os.path.join(args.exp, 'test_results'), 'w') as f:
        for inp, pred in zip(inputs_mapped, prediction_mapped):
            print('{},{}'.format(inp, pred), file=f)
    # precision, recall, fscore = evaluate(test_y, predicted)
    # triv, acc = measure_accuracy(test_y, predicted)
    # print(precision, recall, fscore)
    # print(triv, acc)
    # acc = hist.history['acc']
    # loss = hist.history['loss']
    # plt.plot(range(len(acc)), acc)
    # plt.show()
