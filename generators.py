import numpy as np
import tensorflow as tf
from ops import * 

def vanilla_can(model, z, is_sampler=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = model.output_height, model.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)#8/8
        s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)#4/4 

        z_ = linear(z, model.gf_dim*s_h64*s_w64*16, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, s_h64, s_w64, model.gf_dim*16])
        h0 = lrelu(model.g_bn0(h0, train=is_sampler))

        h1 = model.upsample(
             h0, [-1, s_h32, s_w32, model.gf_dim*16], name='g_h1')
        h1 = lrelu(model.g_bn1(h1, train=is_sampler))

        h2 = model.upsample(
             h1, [-1, s_h16, s_w16, model.gf_dim*8], name='g_h2')
        h2 = lrelu(model.g_bn2(h2, train=is_sampler))

        h3 = model.upsample(
            h2, [-1, s_h8, s_w8, model.gf_dim*4], name='g_h3')
        h3 = lrelu(model.g_bn3(h3, train=is_sampler))

        h4 = model.upsample(
            h3, [-1, s_h4, s_w4, model.gf_dim*2], name='g_h4')
        h4 = lrelu(model.g_bn4(h4, train=is_sampler))

        h5 = model.upsample(
            h4, [-1, s_h2, s_w2, model.gf_dim], name='g_h5')
        h5 = lrelu(model.g_bn5(h5, train=is_sampler))

        h6 = model.upsample(
            h5, [-1, s_h, s_w, model.c_dim], name='g_h6')

        return tf.nn.tanh(h6)

def vanilla_wgan(model, z, y, is_sampler=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = model.output_height, model.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)#8/8
        s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)#4/4

        # project `z` and reshape

        yb = tf.reshape(y, [-1, 1, 1, model.y_dim])
        z_ = concat([z,y],1)

        h0 = lrelu(model.g_bn0(linear(z_, model.gf_dim*16*s_h64*s_w64, 'g_h0_lin'), train=is_sampler))
        h0 = tf.reshape(h0, [-1, s_h64, s_w64, model.gf_dim*16])
        h0 = conv_cond_concat(h0, yb)

        h1 = model.upsample(h0, [-1, s_h32, s_w32, model.gf_dim*16], name='g_h1')
        h1 = lrelu(model.g_bn1(h1, train=is_sampler))
        h1 = conv_cond_concat(h1, yb)

        h2 = model.upsample(h1, [-1, s_h16, s_w16, model.gf_dim*8], name='g_h2')
        h2 = lrelu(model.g_bn2(h2, train=is_sampler))
        h2 = conv_cond_concat(h2, yb)

        h3 = model.upsample(h2, [-1, s_h8, s_w8, model.gf_dim*4], name='g_h3')
        h3 = lrelu(model.g_bn3(h3, train=is_sampler))
        h3 = conv_cond_concat(h3, yb)

        h4 = model.upsample(h3, [-1, s_h4, s_w4, model.gf_dim*2], name='g_h4')
        h4 = lrelu(model.g_bn4(h4, train=is_sampler))
        h4 = conv_cond_concat(h4, yb)

        h5 = model.upsample(h4, [-1, s_h2, s_w2, model.gf_dim], name='g_h5')
        h5 = lrelu(model.g_bn5(h5, train=is_sampler))
        h5 = conv_cond_concat(h5, yb)

        h6 = model.upsample(h5, [-1, s_h, s_w, model.c_dim], name='g_h6')
        return tf.nn.tanh(h6)

def vanilla_wgan_no_y(model, z, is_sampler=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = model.output_height, model.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)#8/8
        s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)#4/4

        # project `z` and reshape

        h0 = lrelu(model.g_bn0(linear(z, model.gf_dim*16*s_h64*s_w64, 'g_h0_lin'), train=is_sampler))
        h0 = tf.reshape(h0, [-1, s_h64, s_w64, model.gf_dim*16])

        h1 = model.upsample(h0, [-1, s_h32, s_w32, model.gf_dim*16], name='g_h1')
        h1 = lrelu(model.g_bn1(h1, train=is_sampler))

        h2 = model.upsample(h1, [-1, s_h16, s_w16, model.gf_dim*8], name='g_h2')
        h2 = lrelu(model.g_bn2(h2, train=is_sampler))

        h3 = model.upsample(h2, [-1, s_h8, s_w8, model.gf_dim*4], name='g_h3')
        h3 = lrelu(model.g_bn3(h3, train=is_sampler))

        h4 = model.upsample(h3, [-1, s_h4, s_w4, model.gf_dim*2], name='g_h4')
        h4 = lrelu(model.g_bn4(h4, train=is_sampler))

        h5 = model.upsample(h4, [-1, s_h2, s_w2, model.gf_dim], name='g_h5')
        h5 = lrelu(model.g_bn5(h5, train=is_sampler))

        h6 = model.upsample(h5, [-1, s_h, s_w, model.c_dim], name='g_h6')
        return tf.nn.tanh(h6)


def can_slim(model, z, is_sampler=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = model.output_height, model.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16

        z_ = linear(z_, model.gf_dim*s_h16*s_w16*8, 'g_h0_lin')
        z_ = tf.reshape(z_, [-1, s_h16, s_w, model.gf_dim*8])
 
        h0 = model.upsample(
             z_, [-1, s_h16, s_w16, model.gf_dim*8], name='g_h0')
        h0 = lrelu(model.g_bn0(h0, train=is_sampler))

        h1 = model.upsample(
            h0, [-1, s_h8, s_w8, model.gf_dim*4], name='g_h1')
        h1 = lrelu(model.g_bn1(h1, train=is_sampler))

        h2 = model.upsample(
            h1, [-1, s_h4, s_w4, model.gf_dim*2], name='g_h2')
        h2 = lrelu(model.g_bn2(h2, train=is_sampler))

        h3 = model.upsample(
            h2, [-1, s_h2, s_w2, model.gf_dim], name='g_h3')
        h3 = lrelu(model.g_bn3(h3, train=is_sampler))

        h4 = model.upsample(
            h3, [-1, s_h, s_w, model.c_dim], name='g_h4')

        return tf.nn.tanh(h4)

def wgan_slim(model, z, y, is_sampler=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = model.output_height, model.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16

        # project `z` and reshape
        yb = tf.reshape(y, [-1, 1, 1, model.y_dim])
        z = concat([z_,y],1)
        z_ = linear(z, model.gf_dim*s_h16*s_w16*8, 'g_h0_lin')
        z_ = tf.reshape(z_, [-1, s_h16, s_w16, model.gf_dim*8]) 

        h0 = model.upsample(z_, [-1, s_h16, s_w16, model.gf_dim*8], name='g_h0')
        h0 = lrelu(model.g_bn0(h0, train=is_sampler))
        h0 = conv_cond_concat(h0, yb)

        h1 = model.upsample(h0, [-1, s_h8, s_w8, model.gf_dim*4], name='g_h1')
        h1 = lrelu(model.g_bn1(h1, train=is_sampler))
        h1 = conv_cond_concat(h1, yb)

        h2 = model.upsample(h1, [-1, s_h4, s_w4, model.gf_dim*2], name='g_h2')
        h2 = lrelu(model.g_bn2(h2, train=is_sampler))
        h2 = conv_cond_concat(h2, yb)

        h3 = model.upsample(h2, [-1, s_h2, s_w2, model.gf_dim], name='g_h3')
        h3 = lrelu(model.g_bn3(h3, train=is_sampler))
        h3 = conv_cond_concat(h3, yb)

        h4 = model.upsample(h3, [-1, s_h, s_w, model.c_dim], name='g_h4')
        return tf.nn.tanh(h4)


def wgan_no_y_slim(model, z, is_sampler=False):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = model.output_height, model.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16

        z_ = linear(z, model.gf_dim*s_h16*s_w16*8, 'g_h0_lin')
        z_ = tf.reshape(z_, [-1, s_h16, s_w16, model.gf_dim*8])
 
        h0 = model.upsample(
             z_, [-1, s_h16, s_w16, model.gf_dim*8], name='g_h0')
        h0 = lrelu(model.g_bn0(h0, train=is_sampler))

        h1 = model.upsample(
            h0, [-1, s_h8, s_w8, model.gf_dim*4], name='g_h1')
        h1 = lrelu(model.g_bn1(h1, train=is_sampler))

        h2 = model.upsample(
            h1, [-1, s_h4, s_w4, model.gf_dim*2], name='g_h2')
        h2 = lrelu(model.g_bn2(h2, train=is_sampler))

        h3 = model.upsample(
            h2, [-1, s_h2, s_w2, model.gf_dim], name='g_h3')
        h3 = lrelu(model.g_bn3(h3, train=is_sampler))

        h4 = model.upsample(
            h3, [-1, s_h, s_w, model.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
