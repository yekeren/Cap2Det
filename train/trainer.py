from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf

from reader import reader
from models import builder

from protos import model_pb2
from protos import pipeline_pb2

from core import training_utils
from train.eval_summary_saver_hook import EvalSummarySaverHook


def _create_model_fn(pipeline_proto, is_chief=True):
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

    # Compute losses. Note: variables created in build_loss are not trainable.

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
    training_hooks = []

    if tf.estimator.ModeKeys.TRAIN == mode:

      train_config = pipeline_proto.train_config

      # Create the optimizer.

      learning_rate = train_config.learning_rate
      global_step = tf.train.get_or_create_global_step()

      if train_config.HasField('learning_rate_decay'):
        learning_rate = tf.train.exponential_decay(
            learning_rate,
            global_step,
            train_config.learning_rate_decay.decay_steps,
            train_config.learning_rate_decay.decay_rate,
            staircase=train_config.learning_rate_decay.staircase)
      tf.summary.scalar('loss/learning_rate', learning_rate)

      optimizer = training_utils.build_optimizer(
          train_config.optimizer, learning_rate=learning_rate)

      # Setup the replicas_hook for the SyncReplicasOptimizer.

      if train_config.sync_replicas:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer, replicas_to_aggregate=4)
        sync_replicas_hook = optimizer.make_session_run_hook(is_chief)
        training_hooks.append(sync_replicas_hook)

      # Enable MovingAverageOptimizer if specified.

      if train_config.HasField('moving_average_decay'):
        optimizer = tf.contrib.opt.MovingAverageOptimizer(
            optimizer, average_decay=train_config.moving_average_decay)

      # Apply gradient multipliers.

      trainable_variables = []
      gradient_multipliers = {}
      for var in variables_to_train:
        add_to_trainable_variables = True

        for multiplier in train_config.gradient_multiplier:
          if var.op.name.startswith(multiplier.scope):
            if var.op.name in gradient_multipliers:
              tf.logging.warn('Override gradient multiplier: %s', var.op.name)
            gradient_multipliers[var.op.name] = multiplier.multiplier
            if multiplier.multiplier > 0:
              add_to_trainable_variables = True
            else:
              add_to_trainable_variables = False

        # Add to trainable variables.
        if add_to_trainable_variables:
          trainable_variables.append(var)
          tf.logging.info('Variable to train: %s, %s', var.op.name,
                          var.get_shape())
        elif var.op.name in gradient_multipliers:
          del gradient_multipliers[var.op.name]

      tf.logging.info('Apply gradient multipliers: \n%s',
                      json.dumps(gradient_multipliers, indent=2))

      def transform_grads_fn(grads):
        if gradient_multipliers:
          grads = tf.contrib.training.multiply_gradients(
              grads, gradient_multipliers)
        if train_config.HasField('max_gradient_norm'):
          grads = tf.contrib.training.clip_gradient_norms(
              grads, max_norm=train_config.max_gradient_norm)
        return grads

      # The train_op is required for mode `TRAIN`.

      train_op = tf.contrib.training.create_train_op(
          total_loss,
          optimizer,
          variables_to_train=trainable_variables,
          transform_grads_fn=transform_grads_fn,
          summarize_gradients=True)

      if train_config.HasField('moving_average_decay'):
        scaffold = tf.train.Scaffold(
            saver=optimizer.swapping_saver(), copy_from_scaffold=scaffold)

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
        training_hooks=training_hooks,
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

  # eval_hooks = [
  #     EvalSummarySaverHook(output_dir=pipeline_proto.model_dir + '/eval')
  # ]
  eval_hooks = None
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

  run_config = tf.estimator.RunConfig(
      save_summary_steps=train_config.save_summary_steps,
      save_checkpoints_steps=train_config.save_checkpoints_steps,
      session_config=session_config,
      keep_checkpoint_max=train_config.keep_checkpoint_max,
      log_step_count_steps=train_config.log_step_count_steps)

  model_fn = _create_model_fn(pipeline_proto, is_chief=run_config.is_chief)

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
