import tensorflow as tf
import argparse
import numpy as np
from copy import deepcopy
import time
import sys, os

from utils import TextLoader

class Model():  
    def __init__(self, config):
        self._num_steps = config.num_steps
        self._embed_size = config.embed_size
        self._batch_size = config.batch_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_layers
        self._rnn_type = config.rnn_type
        self._lr = config.learning_rate
        self._anneal_rate = config.anneal_rate
        self._num_epochs = config.num_epochs
        self._variable_scope = config.variable_scope

        os.makedirs(config.result_path)
        self._save_config(config.result_path)
        self._models_dir = os.path.join(config.result_path, 'models')

        self._text_loader = TextLoader(batch_size=config.batch_size, seq_length=config.num_steps)
        self._vocab_size = self._text_loader.vocab_size

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=sess_config)

        self._inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self._num_steps))
        self._labels_placeholder = tf.placeholder(tf.int32, shape=(None, self._num_steps))

        with tf.variable_scope(self._variable_scope):
            inputs = self._add_embedding()
            rnn_outputs = self._add_model(inputs)
            self.outputs = self._add_projection(rnn_outputs)
        
        self._predictions = tf.nn.softmax(tf.cast(self.outputs, 'float64'))
        self._cross_entropy = self._add_cross_entropy_op(self.outputs)
        self._train_step = self._add_training_op()

    def _save_config(self, result_dir):
        config_path = os.path.join(config.result_path, 'config')
        f = open(config_path, "w")
        f.write(self.__dict__)
        f.close()

    def _add_embedding(self):
        embeddings = tf.get_variable(name='embeddings', shape=[self._vocab_size, self._embed_size])
        inputs = tf.nn.embedding_lookup(embeddings, self._inputs_placeholder)
        batch_mean, batch_var = tf.nn.moments(inputs, [0,1])
        inputs_bn = (inputs - batch_mean)/tf.sqrt(batch_var + 1e-3)
        scale = tf.get_variable(name='scale', shape=[self._embed_size], initializer=tf.constant_initializer(1.0))
        shift = tf.get_variable(name='shift', shape=[self._embed_size], initializer=tf.constant_initializer(0.0))
        inputs = scale*inputs_bn + shift
        return inputs

    def _add_model(self, inputs):
        if self._rnn_type == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif self._rnn_type == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif self._rnn_type == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell

        with tf.variable_scope('RNN') as scope:
            cell = tf.contrib.rnn.MultiRNNCell([cell_fn(self._hidden_size) for _ in range(self._num_layers)])
       
        self._initial_state = cell.zero_state(self._batch_size, tf.float32)
        inputs = tf.unstack(inputs, axis=1)
        rnn_outputs, self._final_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self._initial_state)
        return rnn_outputs

    def _add_projection(self, rnn_outputs):
        with tf.variable_scope('projection') as scope:
            proj_U = tf.get_variable(name='W', shape=[self._hidden_size, self._vocab_size])
            proj_b = tf.get_variable(name='biases', shape=[self._vocab_size], initializer=tf.constant_initializer(0.0))

        concat_rnn_outputs = tf.concat(rnn_outputs, 0)
        concat_outputs = tf.matmul(concat_rnn_outputs, proj_U) + proj_b
        outputs = tf.split(concat_outputs, len(rnn_outputs), axis=0)
        outputs = tf.stack(outputs, axis=1)
        return outputs

    def _add_cross_entropy_op(self, output):
        targets = self._labels_placeholder
        weights = tf.ones([self._batch_size,  self._num_steps], dtype=tf.float32)
        cross_entropy_loss = tf.contrib.seq2seq.sequence_loss(output, targets, weights)
        return cross_entropy_loss

    def _add_training_op(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self._cross_entropy, params)
        optimizer = tf.train.AdamOptimizer(self._lr)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)
        return train_op

    def _run_epoch(self, verbose=10):
        self._lr = self._lr*self._anneal_rate
        total_steps = sum(1 for x in self._text_loader.data_iterator())
        total_loss = []
        state = self._sess.run(self._initial_state)
 
        for step, (x, y) in enumerate(self._text_loader.data_iterator()):
            feed_dict = {self._inputs_placeholder: x, self._labels_placeholder: y, self._initial_state: state}
            loss, state, _ = self._sess.run([self._cross_entropy, self._final_state, self._train_step], feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
				  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss)

    def train(self):
        init = tf.global_variables_initializer()
        self._sess.run(init)
        saver = tf.train.Saver()
        for epoch in xrange(self._num_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            train_loss = self._run_epoch()
            print 'Training loss: {}'.format(train_loss)
            print 'Total time: {}'.format(time.time() - start)

            model_path = os.path.join(self._models_dir, 'model')
            saver.save(self._sess, model_path, write_meta_graph=False, write_state=False)

    def generate_text(self, seed_string, final_len=320):
        seq_len = len(seed_string)
        self._inputs_placeholder = tf.placeholder(tf.int32, shape=(None, seq_len))
        state = session.run(self._initial_state)
        max_iter = final_len - seq_len
        for i in range(max_iter):
            x=np.array([self._text_loader.char2indices[c] for c in seed_string[-seq_len:]])[np.newaxis,:]
            feed_dict = {self._inputs_placeholder: x, self._initial_state: state}
            state, preds = self._sess.run([self._final_state, self._predictions[-1]], feed_dict=feed_dict)
            preds = np.reshape(preds/np.sum(preds), [-1, preds.shape[1]])
            next_char = np.random.choice(self._text_loader.chars, p=preds)
            seed_string = seed_string + next_char
        return seed_string

def add_arguments(parser):
    parser.add_argument('--result-path', type=str, help='path to results directory', required=True)
    parser.add_argument('--num-steps', default=8, type=int, help='RNN sequence length')
    parser.add_argument('--embed-size', default=48, type=int, help='size of character embedding')
    parser.add_argument('--hidden-size', default=512, type=int, help='size of hidden state of RNN')
    parser.add_argument('--num-layers', default=2, type=int, help='number of layers of RNNs')
    parser.add_argument('--rnn-type', default='rnn', choices=['rnn', 'lstm', 'gru'], type=str, help='type of RNN to use')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='learning rate for adam optimizer')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size to use for training')
    parser.add_argument('--anneal-rate', default=0.97, type=float, help='rate by which to decrease the learning rate every epoch')
    parser.add_argument('--num-epochs', default=50, type=int, help='number of epochs to train for')
    parser.add_argument('--variable-scope', default='RNNLM', type=str, help='variable scope of char-rnn')
    return parser

def check_config(config):
    assert not(os.path.exists(config.result_path)), "result dir already exists!"
    assert config.num_steps > 0
    assert config.embed_size > 0
    assert config.hidden_size > 0
    assert config.num_layers > 0
    assert config.learning_rate > 0
    assert config.batch_size > 0
    assert config.anneal_rate > 0
    assert config.num_epochs > 0

def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    config = parser.parse_args()
    check_config(config)

    model = Model(config)
    model.train()

    starting_text = 'ethics is a basic foundation of all that'
    while starting_text:
        output = model.generate_text(starting_text)
        print(output)
        starting_text = raw_input('> ')

if __name__ == "__main__":
    main()
