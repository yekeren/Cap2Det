from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from reader import reader
from models import builder

from protos import model_pb2
from protos import pipeline_pb2

from core import training_utils
from eval_summary_saver_hook import EvalSummarySaverHook


def _create_model_fn(pipeline_proto):
  """Creates a callable that build the model.

  Args:
    pipeline_proto: an instance of pipeline_pb2.Pipeline.

  Returns:
    model_fn: a callable that takes [features, labels, mode, params] as inputs.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  def _model_fn(features, labels, mode, params):
    """
    Args:
      features: a dict mapping from names to tensors, denoting the features.
      labels: a dict mapping from names to tensors, denoting the labels.
      mode: mode parameter required by the estimator.
      params: additional parameters used for creating the model.

    Returns:
      an instance of EstimatorSpec.
    """
    is_training = (tf.estimator.ModeKeys.TRAIN == mode)
    tf.logging.info("Current mode is %s, is_training=%s", mode, is_training)

    model = builder.build(pipeline_proto.model, is_training)
    predictions = model.build_prediction(features)

    # Get scaffold and variables_to_train.

    scaffold = model.get_scaffold()
    variables_to_train = model.get_variables_to_train()

    # Compute losses.

    losses = model.build_loss(predictions, examples=features)
    for name, loss in losses.items():
      tf.losses.add_loss(loss)
      tf.summary.scalar('loss/' + name, loss)

    for loss in tf.losses.get_regularization_losses():
      tf.summary.scalar(
          "loss/regularization/" + '/'.join(loss.op.name.split('/')[:2]), loss)

    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    train_op = None
    eval_metric_ops = None

    if tf.estimator.ModeKeys.TRAIN == mode:

      # The train_op is required for mode `TRAIN`.

      optimizer = training_utils.build_optimizer(
          pipeline_proto.train_config.optimizer,
          learning_rate=pipeline_proto.train_config.learning_rate)
      train_op = tf.contrib.training.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=variables_to_train,
          summarize_gradients=True)

    elif tf.estimator.ModeKeys.EVAL == mode:

      # The eval_metric_ops is optional for mode `EVAL`.

      eval_metric_ops = model.build_evaluation(predictions, examples=features)

    elif tf.estimator.ModeKeys.PREDICT == mode:

      # The predictions is required for mode `PREDICT`.

      predictions.update(features)
      predictions.update({'summary': tf.summary.merge_all()})

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold)

  return _model_fn


def create_train_and_evaluate(pipeline_proto):
  """Creates a callable to train and evaluate.

  Args:
    pipeline_proto: an instance of pipeline_pb2.Pipeline.

  Returns:
    a callable to train and evalute.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  # Create train_spec.

  train_config = pipeline_proto.train_config
  train_input_fn = reader.get_input_fn(pipeline_proto.train_reader)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=train_config.max_steps)

  # Create eval_spec.

  eval_config = pipeline_proto.eval_config
  eval_input_fn = reader.get_input_fn(pipeline_proto.eval_reader)

  eval_hooks = [
      EvalSummarySaverHook(output_dir=pipeline_proto.model_dir + '/eval')
  ]
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=eval_config.steps,
      hooks=eval_hooks,
      start_delay_secs=eval_config.start_delay_secs,
      throttle_secs=eval_config.throttle_secs)

  # Set session config.

  session_config = tf.ConfigProto()
  session_config.allow_soft_placement = True
  session_config.gpu_options.allow_growth = True

  # Create estimator.

  model_fn = _create_model_fn(pipeline_proto)

  run_config = tf.estimator.RunConfig(
      save_summary_steps=train_config.save_summary_steps,
      save_checkpoints_steps=train_config.save_checkpoints_steps,
      session_config=session_config,
      keep_checkpoint_max=train_config.keep_checkpoint_max,
      log_step_count_steps=train_config.log_step_count_steps)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=pipeline_proto.model_dir, config=run_config)

  # Train and evaluate.

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def predict(pipeline_proto, checkpoint_path=None, yield_single_examples=False):
  """Creates a callable to train and evaluate.

  Args:
    pipeline_proto: an instance of pipeline_pb2.Pipeline.
    yield_single_examples: If true, yield single examples.

  Yields:
    example: The prediction result.
  """
  if not isinstance(pipeline_proto, pipeline_pb2.Pipeline):
    raise ValueError('pipeline_proto has to be an instance of Pipeline.')

  predict_input_fn = reader.get_input_fn(pipeline_proto.eval_reader)

  # Create estimator.

  model_fn = _create_model_fn(pipeline_proto)

  session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

  run_config = tf.estimator.RunConfig(session_config=session_config)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=pipeline_proto.model_dir, config=run_config)

  # Predict results.

  for example in estimator.predict(
      input_fn=predict_input_fn,
      checkpoint_path=checkpoint_path,
      yield_single_examples=yield_single_examples):
    yield example
