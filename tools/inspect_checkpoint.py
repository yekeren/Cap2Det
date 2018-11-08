import tensorflow as tf

slim = tf.contrib.slim

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_string('checkpoint_path', '', 'Path to the checkpoint file.')

FLAGS = flags.FLAGS


def main(_):
  ckpt_reader = tf.train.NewCheckpointReader(FLAGS.checkpoint_path)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  for var, shape in ckpt_vars_to_shape_map.items():
    tf.logging.info('%s: %s' % (var, shape))


if __name__ == '__main__':
  tf.app.run()
