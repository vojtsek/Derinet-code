import csv
import numpy as np
import editdistance
import pickle
import io
# make pairs from windows of size 10
# -> l1,l2,deriv?

import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii



def get_all_bigrams(word):
    bigrams = set()
    for w1,w2 in zip(word, word[1:]):
        bigr = w1+w2
        unaccented = remove_accents(bigr)
        bigrams.add(unaccented)
    return bigrams

def create_features(dato):
    pass

def get_halves(w1, w2):
    i = 0
    for i, c in enumerate(zip(w1, w2)):
        if c[0] != c[1]:
            break
    suff1 = w1[i:]
    suff2 = w2[i:]
    if suff1 > suff2:
        tmp = suff1
        suff1 = suff2
        suff2 = tmp
    return w1[:i], (suff1, suff2)


class DataSubset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_data(self):
        return self.X, self.y

    def get_data_as_chars(self):
        pass

class Dataset:
    def __init__(self, fn, test_size=0.2, as_chars=False):
        self.as_chars = as_chars
        suffix_pairs = {}
        SUFFIX_COUNT = 100
        self.dst_from_parents = []
        self.chars2ints = {}
        self.sentence_lens = []
        self.bigrams = set()
        # read and analyze the data
        with io.open(fn, "r", encoding="utf-8") as dest_f:
            data_iter = csv.reader(dest_f,
                                   delimiter='\t',
                                   quotechar='"')
            data = [data for data in data_iter]
        data = np.asarray(data)
        pairs = []
        for n, line in enumerate(data):
            for i in range(-5, 6):
                if i == 0:
                    continue
                idx = n + i
                if idx < 0 or idx >= len(data):
                    continue
                first = line
                second = data[idx, :]
                # for bigr in get_all_bigrams(first[1]):
                #     self.bigrams.add(bigr)
                edge = line[0] == data[idx, -1]
                if edge:
                    self.dst_from_parents.append(np.abs(int(line[0]) - int(data[idx, 0])))
                _, (suff1, suff2) = get_halves(first[1], second[1])
                pairs.append((first, second, edge))
                try:
                    suffix_pairs[(suff1, suff2)] += 1
                except:
                    suffix_pairs[(suff1, suff2)] = 1

        d = sorted([(y, x) for x, y in suffix_pairs.items()], reverse=True)
        suffix_features = {y[1]: x for x, y in enumerate(d[:SUFFIX_COUNT])}

        # construct feature vector
        X = []
        y = []
        children = []
        parents = []
        max_token_length = 0
        self.chars2ints['UNK'] = 0
        self.chars2ints['@'] = 1
        self.chars2ints['&'] = len(self.chars2ints)
        self.chars2ints['#'] = len(self.chars2ints)
        for datapoint in pairs:
            first, second, edge = datapoint
            if not as_chars:
                fv = np.zeros((SUFFIX_COUNT + 1 + 6))
                common_prefix, (suff1, suff2) = get_halves(first[1], second[1])
                try:
                    idx = suffix_features[(suff1, suff2)]
                    fv[idx] = 1
                except:
                    fv[SUFFIX_COUNT] = 1  # other
                fv[SUFFIX_COUNT + 1] = len(common_prefix)
                fv[SUFFIX_COUNT + 2] = editdistance.eval(first[1], second[1])
                fv[SUFFIX_COUNT + 3] = str.isupper(first[1][0])
                fv[SUFFIX_COUNT + 4] = str.isupper(second[1][0])
                fv[SUFFIX_COUNT + 5] = ord(first[3])
                fv[SUFFIX_COUNT + 6] = ord(second[3])
                X.append(fv)
            else:
                token_length = max(len(first[1]), len(second[1])) + 2
                max_token_length = max(max_token_length, token_length)
                self.sentence_lens.append(token_length)
                if edge:
                    child, parent = self.embed(first[1], second[1], edge)
                    children.append(child)
                    parents.append(parent)
            y.append(int(edge))
        if not as_chars:
            X_tmp = np.array(X)
        else:
            self.max_token_length = max_token_length
            parents_tmp = np.zeros((len(parents), max_token_length))
            children_tmp = np.zeros((len(children), max_token_length))

            # shape (no_examples, max_token_length)
            for i, rec in enumerate(parents):
                parents_tmp[i,:len(rec)] = rec
            for i, rec in enumerate(children):
                children_tmp[i,:len(rec)] = rec

            # self.parents = self.transform2onehot(parents_tmp, len(self.chars2ints))
            # self.children = self.transform2onehot(children_tmp, len(self.chars2ints))
            self.parents = parents_tmp
            self.children = children_tmp


        y = np.array(y)
        # X = X_tmp
        self.dataset = (X, y)
        # with open('dataset-subset.dump', 'wb') as f:
        #     pickle.dump(dataset, f)
        if self.as_chars:
            test_size = int(len(self.parents) * test_size)
        else:
            test_size = int(len(X) * test_size)

        self.sentence_lens = np.array(self.sentence_lens)
        self.number_tokens = len(self.chars2ints)
        self.int2chars = { y:x for x, y in self.chars2ints.items()}
        if self.as_chars:
            self.train_X, self.train_y, self.lens_train = self.children[test_size:], self.parents[test_size:], self.sentence_lens[test_size:]
            self.test_X, self.test_y, self.lens_test = self.children[:test_size], self.parents[:test_size], self.sentence_lens[:test_size]
        else:
            self.train_X, self.train_y, self.lens_train = X[test_size:], y[test_size:], self.sentence_lens[test_size:]
            self.test_X, self.test_y, self.lens_test = X[:test_size], y[:test_size], self.sentence_lens[:test_size]
        self.perm = np.random.permutation(len(self.train_X))
        self.current_idx = 0

    def embed(self, v1, v2, is_edge):
        result1 = []
        result2 = []
        for v in v1:
            if v not in self.chars2ints:
                # 0, 1 reserved for separator and UNK
                self.chars2ints[v] = len(self.chars2ints)
            result1.append(self.chars2ints[v])
        # result1.append(0)
        if is_edge:
            for v in v2:
                if v not in self.chars2ints:
                    # 0, 1 reserved for separator and UNK
                    self.chars2ints[v] = len(self.chars2ints)
                result2.append(self.chars2ints[v])
        result2.append(self.chars2ints['#'])
        return result1, result2

    def transform2onehot(self, data, new_dim):
        data_shape = data.shape
        one_hot_data = np.zeros((data_shape[0], data_shape[1], new_dim))
        for i, line in enumerate(data):
            for j, idx in enumerate(line):
                one_hot_data[i,j,(int(idx)-1)] = 1
        return one_hot_data

    def has_next(self):
        if self.current_idx + 10 >= len(self.train_X):
            return False
        return True

    def get_test(self):
        return self.test_X, self.test_y, self.lens_test

    def get_train(self):
        return self.train_X, self.train_y, self.lens_train

    def next_batch(self, bs=10):
        batch = self.train_X[self.perm[self.current_idx:(self.current_idx + bs)], :],\
                self.train_y[self.perm[self.current_idx:(self.current_idx + bs)]],\
                self.lens_train[self.perm[self.current_idx:(self.current_idx + bs)]]
        self.current_idx += bs
        return batch

    def reset(self):
        self.current_idx = 0
        self.perm = np.random.permutation(len(self.train_X))

    def get_string_from_ids(self, sequence):
        return ''.join((self.int2chars[idx] for idx in sequence))

if __name__ == '__main__':
    d = Dataset('data/derismall.tsv', as_chars=True)
    d.next_batch()
    # d.observe()


"""
 featury    - spolecna zakonceni - odseknout prefix, seradit podle poctu vyskytuderivaci
            - delka prefixu
            - Lev. Distance
            - per Lemma slovni druh,
"""
