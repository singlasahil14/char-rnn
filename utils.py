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

        print("reading text file")
        text = open(input_file).read()
        print('corpus length:', len(text))
        chars = sorted(list(set(text)))
        chars.insert(0, "\0")
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
        self.vocab_size = len(self.char2indices)
        self.tensor = np.load(self.tensor_file)
        self.indices2char = {v: k for k, v in self.char2indices.iteritems()}

    def get_batches(self):
        tensor = self.tensor
        seq_length = self.seq_length

        c_in_dat = [[tensor[i+n] for i in xrange(0, len(tensor)-1-seq_length, seq_length)] for n in range(seq_length)]
        c_out_dat = [[tensor[i+n] for i in xrange(1, len(tensor)-seq_length, seq_length)] for n in range(seq_length)]

        xs = [np.stack(c[:-2]) for c in c_in_dat]
        ys = [np.stack(c[:-2]) for c in c_out_dat]

        x_rnn=np.stack(xs, axis=1)
        y_rnn=np.expand_dims(np.stack(ys, axis=1), -1)

        print(x_rnn.shape)
        print(y_rnn.shape)

t = TextLoader()
t.get_batches()
