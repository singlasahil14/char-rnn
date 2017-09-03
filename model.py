import tensorflow as tf
import argparse
import numpy as np
from copy import deepcopy
import time
import sys

from utils import TextLoader

class Model():
  
    def __init__(self, config):
        self._text_loader = TextLoader(batch_size=config.batch_size, seq_length=config.num_steps)
        self._vocab_size = self._text_loader.vocab_size
        self._num_steps = config.num_steps
        self._embed_size = config.embed_size
        self._batch_size = config.batch_size
        self._hidden_size = config.hidden_size
        self._num_layers = config.num_layers
        self._rnn_type = config.rnn_type
        self._lr = config.learning_rate
        self._anneal_rate = config.anneal_rate
        self._add_placeholders()
        inputs = self._add_embedding()
        rnn_outputs = self._add_model(inputs)
        self.outputs = self._add_projection(rnn_outputs)
        self.predictions = tf.nn.softmax(tf.cast(self.outputs, 'float64'))

        self.calculate_loss = self._add_loss_op(self.outputs)
        self.train_step = self._add_training_op(self.calculate_loss)

    def _add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self._num_steps))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self._num_steps))
    
    def _add_embedding(self):
        embeddings = tf.get_variable(name='embeddings', shape=[self._vocab_size, self._embed_size])
        inputs = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        batch_mean, batch_var = tf.nn.moments(inputs, [0,1])
        inputs_bn = (inputs - batch_mean)/tf.sqrt(batch_var + 1e-3)
        scale = tf.get_variable(name='scale', shape=[self._embed_size],
                                initializer=tf.constant_initializer(1.0))
        shift = tf.get_variable(name='shift', shape=[self._embed_size], 
                                initializer=tf.constant_initializer(0.0))
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
       
        self.initial_state = cell.zero_state(self._batch_size, tf.float32)
        inputs = tf.unstack(inputs, axis=1)
        rnn_outputs, self.final_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self.initial_state)
        return rnn_outputs

    def _add_projection(self, rnn_outputs):
        with tf.variable_scope('projection') as scope:
            proj_U = tf.get_variable(name='W', shape=[self._hidden_size, self._vocab_size])
            proj_b = tf.get_variable(name='biases', shape=[self._vocab_size],
                                     initializer=tf.constant_initializer(0.0))

        concat_rnn_outputs = tf.concat(rnn_outputs, 0)
        concat_outputs = tf.matmul(concat_rnn_outputs, proj_U) + proj_b
        outputs = tf.split(concat_outputs, len(rnn_outputs), axis=0)
        outputs = tf.stack(outputs, axis=1)
        return outputs

    def _add_loss_op(self, output):
        targets = self.labels_placeholder
        weights = tf.ones([self._batch_size,  self._num_steps], dtype=tf.float32)
        cross_entropy_loss = tf.contrib.seq2seq.sequence_loss(output, targets, weights)

        return cross_entropy_loss

    def _add_training_op(self, loss):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        optimizer = tf.train.AdamOptimizer(self._lr)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)
        return train_op

    def run_epoch(self, session, train_op=None, verbose=10):
        self._lr = self._lr*self._anneal_rate
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in self._text_loader.data_iterator())
        total_loss = []
        state = session.run(self.initial_state)
 
        for step, (x, y) in enumerate(self._text_loader.data_iterator()):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.inputs_placeholder: x, self.labels_placeholder: y,
                    self.initial_state: state}
            loss, state, _ = session.run(
                  [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
				  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss)

def generate_text(session, scope_name, orig_config, seed_string, final_len=320):
    text_loader = TextLoader(batch_size=orig_config.batch_size, seq_length=orig_config.num_steps)
    seq_len = len(seed_string)

    gen_config = deepcopy(orig_config)
    gen_config.batch_size = 1
    gen_config.num_steps = seq_len
    with tf.variable_scope(scope_name) as scope:
        scope.reuse_variables()
        gen_model = Model(gen_config)
    state = session.run(gen_model.initial_state)
    max_iter = final_len - seq_len
    for i in range(max_iter):
        x=np.array([text_loader.char2indices[c] for c in seed_string[-seq_len:]])[np.newaxis,:]
        feed = {gen_model.inputs_placeholder: x, 
                gen_model.initial_state: state}
        state, preds = session.run([gen_model.final_state, gen_model.predictions[-1]], 
                                    feed_dict=feed)
        preds = np.reshape(preds/np.sum(preds), [-1, preds.shape[1]])
        print(preds.shape)
        next_char = np.random.choice(text_loader.chars, p=preds)
        seed_string = seed_string + next_char
    return seed_string

def generate_sentence(session, scope_name, orig_config, seed_string):
    """Convenice to generate a sentence from the model."""
    return generate_text(session, scope_name, orig_config, seed_string)

def add_arguments(parser):
    parser.add_argument('--num-steps', default=8, type=int, help='RNN sequence length')
    parser.add_argument('--embed-size', default=48, type=int, help='size of character embedding')
    parser.add_argument('--hidden-size', default=512, type=int, help='size of hidden state of RNN')
    parser.add_argument('--num-layers', default=2, type=int, help='number of layers of RNNs')
    parser.add_argument('--rnn-type', default='rnn', choices=['rnn', 'lstm', 'gru'], type=str, help='type of RNN to use')
    parser.add_argument('--learning-rate', default=0.001, type=float, help='learning rate for adam optimizer')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size to use for training')
    parser.add_argument('--anneal-rate', default=0.97, type=float, help='rate by which to decrease the learning rate every epoch')
    parser.add_argument('--max-epochs', default=50, type=int, help='number of epochs to train for')
    return parser

def check_config(config):
    assert config.num_steps > 0
    assert config.embed_size > 0
    assert config.hidden_size > 0
    assert config.num_layers > 0
    assert config.learning_rate > 0
    assert config.batch_size > 0
    assert config.anneal_rate > 0
    assert config.max_epochs > 0

def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    config = parser.parse_args()
    check_config(config)

    # We create the training model and generative model
    scope_name = 'RNNLM'
    with tf.variable_scope(scope_name) as scope:
        model = Model(config)
        # This instructs gen_model to reuse the same variables as the model above

    model_file = './models/char_rnnlm.weights' 
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    session = tf.Session(config=sess_config)
    best_val_pp = float('inf')
    best_val_epoch = 0

    init = tf.global_variables_initializer()
    session.run(init)
    saver = tf.train.Saver()
    for epoch in xrange(config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()
        ###
        train_loss = model.run_epoch(session, train_op=model.train_step)
        print 'Training loss: {}'.format(train_loss)
        saver.save(session, model_file)
        print 'Total time: {}'.format(time.time() - start)

    saver.restore(session, model_file)
    starting_text = 'ethics is a basic foundation of all that'
    while starting_text:
        output = generate_sentence(session, scope_name, 
                          config, seed_string=starting_text)
        print(output)
        starting_text = raw_input('> ')

if __name__ == "__main__":
    main()
