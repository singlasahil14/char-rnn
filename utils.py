import numpy as np
import os
from six.moves import cPickle

class TextLoader():
    def __init__(self, data_dir='nietzsche', batch_size=128, seq_length=8):
        self.data_dir = "data/" + data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        self.input_file = os.path.join(self.data_dir, "input.txt")
        self.vocab_map_file = os.path.join(self.data_dir, "vocab-map.pkl")
        self.tensor_file = os.path.join(self.data_dir, "tensor.npy")

        if not(os.path.exists(self.vocab_map_file) and os.path.exists(self.tensor_file)):
            self.preprocess()
        else:
            self.load_preprocessed()

    def preprocess(self):
        input_file = self.input_file
        vocab_map_file = self.vocab_map_file
        tensor_file = self.tensor_file

        text = open(input_file).read()
        chars = list(set(text))
        chars.insert(0, "\0")
        self.chars = sorted(chars)
        self.vocab_size = len(chars)

        self.char2indices = dict((c, i) for i, c in enumerate(chars))
        self.indices2char = dict((i, c) for i, c in enumerate(chars))
        with open(vocab_map_file, 'wb') as f:
            cPickle.dump(self.char2indices, f)
        self.tensor = np.array(list(map(self.char2indices.get, text)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self):
        with open(self.vocab_map_file, 'rb') as f:
            self.char2indices = cPickle.load(f)
        self.chars = sorted(self.char2indices.keys())
        self.vocab_size = len(self.char2indices)
        self.tensor = np.load(self.tensor_file)
        self.indices2char = {v: k for k, v in self.char2indices.iteritems()}

    def data_iterator(self):
        tensor = self.tensor
        batch_size = self.batch_size
        seq_length = self.seq_length

        data_len = len(tensor)
        batch_len = batch_size * seq_length
        data_len = data_len - (data_len%batch_len) - batch_len
        size_per_batch = data_len//batch_size
        epoch_size = data_len//batch_len

        data = np.zeros([batch_size, size_per_batch + 1], dtype=np.int32)
        for i in range(batch_size):
            data[i] = tensor[size_per_batch * i: size_per_batch * (i + 1) + 1]

        for i in range(epoch_size):
            x = data[:, i * seq_length:(i + 1) * seq_length]
            y = data[:, i * seq_length + 1:(i + 1) * seq_length + 1]
            yield(x, y)
