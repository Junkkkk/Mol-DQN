# """Utility functions and other shared chemgraph code."""
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# from absl import logging
# import tensorflow as tf
# from tensorflow import gfile
#
#
# def read_hparams(filename, defaults):
#   """Reads HParams from JSON.
#   Args:
#     filename: String filename.
#     defaults: HParams containing default values.
#   Returns:
#     HParams.
#   Raises:
#     gfile.Error: If the file cannot be read.
#     ValueError: If the JSON record cannot be parsed.
#   """
#   with gfile.Open(filename) as f:
#     logging.info('Reading HParams from %s', filename)
#     return defaults.parse_json(f.read())
#
#
# def write_hparams(hparams, filename):
#   """Writes HParams to disk as JSON.
#   Args:
#     hparams: HParams.
#     filename: String output filename.
#   """
#   with gfile.Open(filename, 'w') as f:
#     f.write(hparams.to_json(indent=2, sort_keys=Tru separators=(',', ': ')))
# #
# #
# # def learning_rate_decay(initial_learning_rate, decay_steps, decay_rate):
# #   """Initializes exponential learning rate decay.
# #   Args:
# #     initial_learning_rate: Float scalar tensor containing the initial learning
# #       rate.
# #     decay_steps: Integer scalar tensor containing the number of steps between
# #       updates.
# #     decay_rate: Float scalar tensor containing the decay rate.
# #   Returns:
# #     Float scalar tensor containing the learning rate. The learning rate will
# #     automatically be exponentially decayed as global_step increases.
# #   """
# #   with tf.variable_scope('learning_rate_decay'):
# #     learning_rate = tf.train.exponential_decay(
# #         learning_rate=initial_learning_rate,
# #         global_step=tf.train.get_global_step(),
# #         decay_steps=decay_steps,
# #         decay_rate=decay_rate)
# #   tf.summary.scalar('learning_rate', learning_rate)
# #   return learning_ratee,