from __future__ import division
from __future__ import print_function

from keras_models import FeedForward, Seq2Seq, Seq2Classify
from utils import evaluate
import pickle

if __name__ == '__main__':

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size.')
    parser.add_argument('--layer_sizes', default='20,10', type=str, help='Sizes of respective layers; comma separated')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs.')
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
    model = Seq2Classify(num_tokens=dataset.number_tokens, max_len=dataset.max_token_length)
    model.train(train_X, train_y, epochs=1, new_dim=dataset.number_tokens, dataset=dataset)
    # with open('model.bin', 'wb') as f:
    #     pickle.dump(model, f)

    # with open('model.bin', 'rb') as f:
    #     model = pickle.load(f)
    # evaluated = model.evaluate(test_X, test_y)
    for i in range(20):
        predicted = model.predict(test_X[i], dataset.chars2ints, dataset.int2chars)
        print('\n{}\n'.format(predicted[:30]))


