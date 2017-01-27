import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np
from copy import deepcopy
import time
import sys

from utils import TextLoader

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    num_steps = 8
    embed_size = 48
    hidden_size = 512
    num_layers = 2
    model_type = 'lstm'
    lr = 0.001

    batch_size = 128
    anneal_by = 0.97
    max_epochs = 50
    text_loader = TextLoader(batch_size=batch_size, seq_length=num_steps)
    vocab_size = text_loader.vocab_size

class Model():
  
    def __init__(self, config):
        self.config = config
        self.add_placeholders()
        inputs = self.add_embedding()
        rnn_outputs = self.add_model(inputs)
        self.outputs = self.add_projection(rnn_outputs)
  
        # We want to check how well we correctly predict the next word
        # We cast o to float64 as there are numerical issues at hand
        # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as
        # needed to evenly divide
        output = tf.reshape(tf.concat(1, self.outputs), [-1, self.config.vocab_size])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors
        """
        num_steps = self.config.num_steps
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, num_steps))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, num_steps))
    
    def add_embedding(self):
        """Add embedding layer.
        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
	    """
        embed_size = self.config.embed_size
        vocab_size = self.config.vocab_size
        num_steps = self.config.num_steps
        embeddings = tf.get_variable(name='embeddings',
                                     shape=[vocab_size, embed_size])
        inputs = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        batch_mean, batch_var = tf.nn.moments(inputs, [0,1])
        inputs_bn = (inputs - batch_mean)/tf.sqrt(batch_var + 1e-3)
        scale = tf.get_variable(name='scale', 
                                shape=[embed_size],
                                initializer=tf.constant_initializer(1.0))
        shift = tf.get_variable(name='shift', 
                                shape=[embed_size], 
                                initializer=tf.constant_initializer(0.0))
        inputs = scale*inputs_bn + shift
        inputs = tf.unstack(inputs, axis=1)
        return inputs

    def add_model(self, inputs):
        """Creates the char-rnn language model hidden layer
        Args:
          inputs: List of length num_steps, each of whose elements should be
          	     a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each of whose elements should be
                 a tensor of shape (batch_size, hidden_size)
	    """
        batch_size = self.config.batch_size
        embed_size = self.config.embed_size
        hidden_size = self.config.hidden_size
        num_steps = self.config.num_steps
        num_layers = self.config.num_layers
        model_type = self.config.model_type

        if model_type == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif model_type == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif model_type == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell

        with tf.variable_scope('RNN') as scope:
            cell = cell_fn(hidden_size, state_is_tuple=True)
            cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        
        self.initial_state = state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, self.final_state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)
        return rnn_outputs

    def add_projection(self, rnn_outputs):
        """Adds a projection layer.
        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.
        Args:
	      rnn_outputs: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, hidden_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, vocab_size)
        """
        num_steps = self.config.num_steps
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        with tf.variable_scope('projection') as scope:
            proj_U = tf.get_variable(name='W', shape=[hidden_size, vocab_size])
            proj_b = tf.get_variable(name='biases', shape=[vocab_size],
                                     initializer=tf.constant_initializer(0.0))

        concat_rnn_outputs = tf.concat(0, rnn_outputs)
        concat_outputs = tf.matmul(concat_rnn_outputs, proj_U) + proj_b
        outputs = tf.split(0, len(rnn_outputs), concat_outputs)
        return outputs

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.
        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        num_steps = self.config.num_steps
        batch_size = self.config.batch_size
        vocab_size = self.config.vocab_size

        targets = tf.reshape(self.labels_placeholder, [-1])
        weights = tf.ones([batch_size * num_steps], dtype=tf.float32)
        cross_entropy_loss = seq2seq.sequence_loss([output], [targets], [weights], vocab_size)

        return cross_entropy_loss

    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        optimizer = tf.train.AdamOptimizer(self.config.lr)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)
        return train_op

    def run_epoch(self, session, train_op=None, verbose=10):
        config = self.config
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in self.config.text_loader.data_iterator())
        total_loss = []
        state = session.run(self.initial_state)
 
        for step, (x, y) in enumerate(
            self.config.text_loader.data_iterator()):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.inputs_placeholder: x,
                    self.labels_placeholder: y,
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
    """Generate text from the model.
    Args:
        session: tf.Session() object
        scope_name: name of scope in which model exists
        orig_config: A Config() object
        seed_string: Initial text passed to model
        final_len: Final length of output string
    Returns:
        output: List of word idxs
    """
    text_loader = orig_config.text_loader
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
        preds = np.reshape(preds/np.sum(preds), preds.shape[1])
        next_char = np.random.choice(text_loader.chars, p=preds)
        seed_string = seed_string + next_char
    return seed_string

def generate_sentence(session, scope_name, orig_config, seed_string):
    """Convenice to generate a sentence from the model."""
    return generate_text(session, scope_name, orig_config, seed_string)

def test_charRNN():
    config = Config()
  
    # We create the training model and generative model
    scope_name = 'RNNLM'
    with tf.variable_scope(scope_name) as scope:
        model = Model(config)
        # This instructs gen_model to reuse the same variables as the model above

    model_file = './models/char_rnnlm.weights' 
    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0
  
        init = tf.global_variables_initializer()
        session.run(init)
        saver = tf.train.Saver()
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()
            ###
            train_loss = model.run_epoch(
            session, train_op=model.train_step)
            model.config.lr = model.config.lr*model.config.anneal_by
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
    test_charRNN()
