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

def WGAN_loss(model):
    model.g_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)
    model.d_opt = tf.train.AdamOptimizer(learning_rate=model.learning_rate, beta1=0.5)
    
    model.G = model.generator(model.z)
    _, model.D_real, model.D_c, _ = model.discriminator(model.inputs, reuse=False)
    model.sampler = model.sampler(model.z)
    _, model.D_fake, model.D_c_, _ = model.discriminator(model.G, reuse=True)

    model.correct_prediciton = tf.equal(tf.argmax(self.y,1), tf.argmax(self.D_c,1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    model.g_loss = -tf.reduce_mean(model.D_fake)
    model.d_loss = tf.reduce_mean(model.D_fake) - tf.reduce_mean(model.D_real)
    
    epsilon = tf.random_uniform(
        shape=tf.shape(model.D_real),
        minval=0.,
        maxval=1.
    )
    x_hat = model.inputs + epsilon * (model.G - model.inputs)
    D_x_hat = model.discriminator(x_hat, reuse=True)
    grad_D_x_hat = tf.gradients(D_x_hat, [x_hat])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_x_hat), reduction_indices=[-1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    model.d_loss += 10 * gradient_penalty 

    t_vars = tf.trainable_variables()
    model.d_vars = [var for var in t_vars if 'd_' in var.name]
    model.g_vars = [var for var in t_vars if 'g_' in var.name]
    
    g_gradvar = model.g_opt.compute_gradients(
        model.g_loss,
        var_list=model.g_vars,
        colocate_gradients_with_ops=True
    )                                          )
    g_update = model.g_opt.apply_gradients(g_gradvar)
    
    d_gradvar = model.d_opt.compute_gradients(
        model.d_loss,
        var_list=model.d_vars,
        colocate_gradients_with_ops=True
    )
    d_update = model.g_opt.apply_gradients(g_gradvar)
    loss_ops = [model.g_loss, model.d_loss] 
    #TODO summaries
    return g_update, d_update, loss_ops



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

def resizeconv(input_, output_shape,
		k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
		name="resconv"):
  with tf.variable_scope(name):
    
    resized = tf.image.resize_nearest_neighbor(input_,((output_shape[1]-1)*d_h + k_h-4, (output_shape[2]-1)*d_w + k_w-4))
    #The 4 is because of same padding in tf.nn.conv2d.
    w = tf.get_variable('w', [k_h, k_w, resized.get_shape()[-1], output_shape[-1]],
		initializer=tf.truncated_normal_initializer(stddev=stddev))
    resconv = tf.nn.conv2d(resized, w, strides=[1, d_h, d_w, 1], padding='SAME')
    biases = tf.get_variable('biases', output_shape[-1], initializer=tf.constant_initializer(0.0))
    
    return tf.nn.bias_add(resconv, biases)
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    static_shape = input_.get_shape().as_list()
    dyn_input_shape = tf.shape(input_)
    batch_size = dyn_input_shape[0]
    out_h = output_shape[1]
    out_w = output_shape[2]
    out_shape = tf.stack([batch_size, out_h, out_w, output_shape[-1]])

    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
     
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=out_shape,
                strides=[1, d_h, d_w, 1])
    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.nn.bias_add(deconv, biases)
    #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))
    deconv.set_shape([None] + output_shape[1:])
    return deconv

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
