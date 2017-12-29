import numpy as np
import tensorflow as tf
from ops import conv_cond_concat
from ops import *

def vanilla_can(model, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        
        h0 = lrelu(conv2d(image, model.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID'))
        h1 = lrelu(model.d_bn1(conv2d(h0, model.df_dim*2, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(model.d_bn2(conv2d(h1, model.df_dim*4, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        h3 = lrelu(model.d_bn3(conv2d(h2, model.df_dim*8, k_h=4, k_w=4, name='d_h3_conv', padding='VALID')))
        h4 = lrelu(model.d_bn4(conv2d(h3, model.df_dim*16, k_h=4, k_w=4, name='d_h4_conv', padding='VALID')))
        shape = np.product(h4.get_shape()[1:].as_list())
        h5 = tf.reshape(h4, [-1, shape])
        #linear layer to determine if the image is real/fake
        r_out = linear(h5, 1, 'd_ro_lin')

        #fully connected layers to classify the image into the different styles.
        h6 = lrelu(linear(h5, 1024, 'd_h6_lin'))
        h7 = lrelu(linear(h6, 512, 'd_h7_lin'))
        c_out = linear(h7, model.y_dim, 'd_co_lin')
        c_softmax = tf.nn.softmax(c_out)

        return tf.nn.sigmoid(r_out), r_out, c_softmax, c_out

def wgan_cond(model, image, y, reuse=False):
            #no batchnorm for WGAN GP
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        
        yb = tf.reshape(y, [-1, 1, 1, model.y_dim])
        image_ = conv_cond_concat(image, yb)
        h0 = lrelu(layer_norm(conv2d(image_, model.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID')))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(layer_norm(conv2d(h0, model.df_dim*4, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(layer_norm(conv2d(h1, model.df_dim*8, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        h2 = conv_cond_concat(h2, yb)
        h3 = lrelu(layer_norm(conv2d(h2, model.df_dim*16, k_h=4, k_w=4, name='d_h3_conv', padding='VALID')))
        h3 = conv_cond_concat(h3, yb)
        h4 = lrelu(layer_norm(conv2d(h3, model.df_dim*32, k_h=4, k_w=4, name='d_h4_conv', padding='VALID')))
        h4 = conv_cond_concat(h4, yb)
        h5 = lrelu(layer_norm(conv2d(h4, model.df_dim*32, k_h=4, k_w=4, name='d_h5_conv', padding='VALID')))

        shape = np.product(h5.get_shape()[1:].as_list())
        h5 = tf.reshape(h5, [-1, shape])
        h5 = concat([h5,y],1)

        r_out = linear(h5, 1, 'd_ro_lin')
        return r_out

def vanilla_wgan(model, image, reuse=False):

    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        
        h0 = lrelu(layer_norm(conv2d(image, model.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID')))
        h1 = lrelu(layer_norm(conv2d(h0, model.df_dim*4, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(layer_norm(conv2d(h1, model.df_dim*8, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        h3 = lrelu(layer_norm(conv2d(h2, model.df_dim*16, k_h=4, k_w=4, name='d_h3_conv', padding='VALID')))
        h4 = lrelu(layer_norm(conv2d(h3, model.df_dim*32, k_h=4, k_w=4, name='d_h4_conv', padding='VALID')))
        h5 = lrelu(layer_norm(conv2d(h4, model.df_dim*32, k_h=4, k_w=4, name='d_h5_conv', padding='VALID')))

        shape = np.product(h5.get_shape()[1:].as_list())
        h5 = tf.reshape(h5, [-1, shape])

        out = linear(h5, 1, 'd_ro_lin')
        return out



def can_slim(model, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = lrelu(model.d_bn0(conv2d(image, model.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID')))
        h1 = lrelu(model.d_bn1(conv2d(h0, model.df_dim*4, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(model.d_bn2(conv2d(h1, model.df_dim*8, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        shape = np.product(h2.get_shape()[1:].as_list())
        h2 = tf.reshape(h2, [-1, shape])
        r_out = linear(h2, 1, 'd_ro_lin')

        h3 = lrelu(linear(h2, 1024, 'd_h6_lin'))
        h4 = lrelu(linear(h3, 512, 'd_h7_lin'))
        c_out = linear(h4, model.y_dim, 'd_co_lin')
        c_softmax = tf.nn.softmax(c_out)

        return tf.nn.sigmoid(r_out), r_out, c_softmax, c_out

def wgan_slim_cond(model, image, y, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        yb = tf.reshape(y, [-1, 1, 1, model.y_dim])
        image_ = conv_cond_concat(image, yb)
        h0 = lrelu(layer_norm(conv2d(image_, model.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID')))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(layer_norm(conv2d(h0, model.df_dim*4, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(layer_norm(conv2d(h1, model.df_dim*8, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        h2 = conv_cond_concat(h2, yb)

        shape = np.product(h2.get_shape()[1:].as_list())
        h3 = tf.reshape(h2, [-1, shape])
        h3 = concat([h3,y],1)

        r_out = linear(h3, 1, 'd_ro_lin')
        return r_out

def wgan_slim(model, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = lrelu(layer_norm(conv2d(image, model.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID')))
        h1 = lrelu(layer_norm(conv2d(h0, model.df_dim*4, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(layer_norm(conv2d(h1, model.df_dim*8, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))

        shape = np.product(h2.get_shape()[1:].as_list())
        h2 = tf.reshape(h2, [-1, shape])

        out = linear(h2, 1, 'd_ro_lin')
        return out
def dcwgan(model, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        h0 = lrelu(conv2d(image, model.df_dim, name='d_h0_conv'))
        h1 = lrelu(layer_norm(conv2d(h0, model.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(layer_norm(conv2d(h1, model.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(layer_norm(conv2d(h2, model.df_dim*8, name='d_h3_conv')))
        shape = np.product(h3.get_shape()[1:].as_list())
        reshaped = tf.reshape(h3, [-1, shape])
        h4 = linear(reshaped, 1, 'd_h4_lin')
        return h4

def dcwgan_cond(model, image, y, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        yb = tf.reshape(y, [-1, 1, 1, model.y_dim])
        x = conv_cond_concat(image, yb)
        h0 = lrelu(conv2d(x, model.df_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(layer_norm(conv2d(h0, model.df_dim*2, name='d_h1_conv')))
        h1 = conv_cond_concat(h1, yb)
        h2 = lrelu(layer_norm(conv2d(h1, model.df_dim*4, name='d_h2_conv')))
        h2 = conv_cond_concat(h2, yb)
        h3 = lrelu(layer_norm(conv2d(h2, model.df_dim*8, name='d_h3_conv')))
        shape = np.product(h3.get_shape()[1:].as_list())
        reshaped = tf.reshape(h3, [-1, shape])
        cond = concat([reshaped,y],1)
        h4 = linear(cond, 1, 'd_h4_lin')
        return h4

