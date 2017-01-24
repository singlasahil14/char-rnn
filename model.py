import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    num_steps = 8
    embed_size = 35
    hidden_size = 512
    num_layers = 3
    vocab_size = 86
    model_type = 'lstm'
    l2 = 0.002
    lr = 0.001

    batch_size = 128
    anneal_by = 1.5
    anneal_after = 10
    max_epochs = 30
    model_dir = 'models'

class Model():
    def __init__(self, config):
        self.config = config

    def add_placeholders(self):
      """Generate placeholder variables to represent the input tensors"""
        self.inputs_placeholder = tf.placeholder(tf.int32, shape = (None, num_steps))
        self.labels_placeholder = tf.placeholder(tf.int32, shape= (None, num_steps))

    def add_embedding(self):
      """Add embedding layer.
      Returns:
        inputs: List of length num_steps, each of whose elements should be
                a tensor of shape (batch_size, embed_size).
      """
        embed_size = self.config.embed_size
        vocab_size = self.config.vocab_size
        embeddings = tf.get_variable(name='embeddings',
                                     shape=[vocab_size, embed_size], 
                                     trainable=True,
                                     dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
        inputs = tf.unstack(inputs, axis=1)
        return inputs

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
        concat_outputs = tf.matmul(concat_rnn_outputs, self.U) + self.b_2
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

        params = tf.trainable_variables()
        l2_loss = tf.add_n([ tf.nn.l2_loss(v) for v in params]) * self.config.l2
        loss = cross_entropy_loss + l2_loss
        return loss

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

        cell = cell_fn(hidden_size, state_is_tuple=True)
        stacked_cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        self.initial_state = state = tf.zeros([batch_size, hidden_size])
        for i in range(num_steps):
            output, state = stacked_cell(inputs[:, i], state)
            rnn_outputs.append(output)
        self.final_state = state
        return rnn_outputs

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
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.apply_gradients(zip(gradients, params))
        return train_op
