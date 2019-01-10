from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class EvalSummarySaverHook(tf.train.SessionRunHook):
  """Saves summaries during eval loop."""

  def __init__(self, output_dir=None, stop_after=1):
    """Initializes a special `SummarySaverHook` to run during evaluations

    Args:
      output_dir: `string`, the directory to save the summaries to.
    """
    self._output_dir = output_dir
    self._global_step_tensor = None
    self._stop_after = stop_after
    self._saves = None

  def begin(self):
    self._global_step_tensor = tf.train.get_or_create_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use SummarySaverHook.")

  def before_run(self, run_context):
    requests = { "global_step": self._global_step_tensor }
    summary_op = tf.get_collection(tf.GraphKeys.SUMMARY_OP)
    if self._saves is None:
      self._saves = 0
    elif self._saves <= self._stop_after and summary_op is not None:
      requests["summary"] = summary_op
    return tf.train.SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    if "summary" in run_values.results:
      global_step = run_values.results["global_step"]
      summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
      for summary in run_values.results["summary"]:
        summary_writer.add_summary(summary, global_step)
      tf.logging.info('Saving eval summaries at %i', global_step)
    self._saves += 1

  def end(self, session=None):
    summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
    summary_writer.flush()
    self._saves = None
