import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d",padding='SAME'):
  with tf.variable_scope(name):
    if padding=='VALID':
      paddings = np.array([[0,0],[1,1],[1,1],[0,0]])
      input_ = tf.pad(input_, paddings)
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    out_shape = [-1] + conv.get_shape()[1:].as_list() 
    conv = tf.reshape(tf.nn.bias_add(conv, biases), out_shape) 
    return conv

def resizeconv(input_, output_dim,
		k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
		name="resconv"):
  with tf.variable_scope(name):
    
    resized = tf.image.resize_nearest_neighbor(input_,((output_dim[1]-1)*d_h + k_h-4, (output_dim[2]-1)*d_w + k_w-4))
    #The 4 is because of same padding in tf.nn.conv2d.
    w = tf.get_variable('w', [k_h, k_w, resized.get_shape()[-1], output_dim[-1]],
		initializer=tf.truncated_normal_initializer(stddev=stddev))
    resconv = tf.nn.conv2d(resized, w, strides=[1, d_h, d_w, 1], padding='SAME')
    biases = tf.get_variable('biases', output_dim[-1], initializer=tf.constant_initializer(0.0))

    resconv = tf.reshape(tf.nn.bias_add(resconv, biases), output_dim)
    return resconv

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    return tf.matmul(input_, matrix) + bias
