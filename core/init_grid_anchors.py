from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import sqrt


def initialize_grid_anchors(scale_list=[0.1, 0.2, 0.35, 0.5, 0.7, 0.9],
                            aspect_ratios_list=[[0.5, 1.0, 2.0],
                                                [0.333333, 0.5, 1.0, 2.0, 3.0],
                                                [0.333333, 0.5, 1.0, 2.0, 3.0],
                                                [0.333333, 0.5, 1.0, 2.0, 3.0],
                                                [0.333333, 0.5, 1.0, 2.0, 3.0],
                                                [0.333333, 0.5, 1.0, 2.0, 3.0]],
                            stride_ratio=0.2):
  """Creates grid anchors.

  Args:
    scales: Scale factor of each resolution.
    aspect_ratios: Aspect ratios of each resolution.
    stride_ratio: How much to move horizontally or vertically.

  Returns:
    A list of anchor boxes in the format of [ymin, xmin, ymax, xmax].
  """
  anchors = []
  for scale, aspect_ratios in zip(scale_list, aspect_ratios_list):
    base_h = base_w = scale
    for aspect_ratio in aspect_ratios:
      # Get box height and width.
      box_h = base_h / sqrt(aspect_ratio)
      box_w = base_w * sqrt(aspect_ratio)

      if box_h > 1.0 or box_w > 1.0:
        continue

      # Estimate the strides ROUGHLY.
      stride_h = box_h * stride_ratio
      stride_w = box_w * stride_ratio

      # Compute the number of grids.
      grids_h = int(np.ceil((1.0 - box_h) / stride_h))
      grids_w = int(np.ceil((1.0 - box_w) / stride_w))

      # Meshgrids.
      if grids_h == 1:
        y_axis = np.array([0.5])
      else:
        y_axis = np.linspace(box_h / 2, 1.0 - box_h / 2, grids_h)

      if grids_w == 1:
        x_axis = np.array([0.5])
      else:
        x_axis = np.linspace(box_w / 2, 1.0 - box_w / 2, grids_w)

      center_h, center_w = np.meshgrid(y_axis, x_axis)
      for c_y, c_x in zip(center_h.flatten(), center_w.flatten()):
        ymin, xmin = c_y - box_h / 2, c_x - box_w / 2
        ymax, xmax = c_y + box_h / 2, c_x + box_w / 2
        anchors.append([ymin, xmin, ymax, xmax])
  return np.array(anchors, dtype=np.float32)
