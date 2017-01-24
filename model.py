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
        cross_entropy = sequence_loss([output], [targets], [weights], vocab_size)
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
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
