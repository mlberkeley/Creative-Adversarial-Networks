from __future__ import division
import os
import time
import math
import random
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from random import shuffle
from ops import *
from utils import *

class Classifier(object):
    
    def __init__(self, sess, input_shape=64, y_dim=27, cf_dim=32,
                 input_fname_pattern='*/*.jpg', checkpoint_dir='checkpoint',
                 learning_rate=1e-4):
        self.sess = sess
        self.y_dim = y_dim
        self.cf_dim = cf_dim
        self.input_shape = input_shape
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.learning_rate=learning_rate
        
        self.bn0 = batch_norm(name='bn0')
        self.bn1 = batch_norm(name='bn1')
        self.bn2 = batch_norm(name='bn2')
        self.bn3 = batch_norm(name='bn3')
        self.bn4 = batch_norm(name='bn4')
        self.build_model()

    def build_model(self):
        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
        image_dims = [self.input_shape, self.input_shape, 3]
        self.inputs = tf.placeholder(tf.float32, [None] + image_dims)
        self.y_logits, self.y_hat = self.forward_pass(image=self.inputs)        
        
        correct_pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_hat,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits, labels=self.y))
        
        self.ambiguity_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits, labels=(1.0/self.y_dim) * tf.ones_like(self.y_logits)))

        self.loss_sum = scalar_summary("c_loss", self.loss)
        self.acc_sum = scalar_summary("accuracy", self.accuracy)
        self.summary = merge_summary([self.loss_sum, self.acc_sum])
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
        self.update = self.opt.minimize(self.loss)
        
        self.saver = tf.train.Saver()

    def forward_pass(self, image):

        h0 = lrelu(self.bn0(conv2d(image, self.cf_dim, name='h0_conv', padding='VALID')))
        h1 = lrelu(self.bn1(conv2d(h0, self.cf_dim*2, name='h1_conv', padding='VALID')))
        h2 = lrelu(self.bn2(conv2d(h1, self.cf_dim*4, name='h2_conv', padding='VALID')))
        h3 = lrelu(self.bn3(conv2d(h2, self.cf_dim*8, name='h3_conv', padding='VALID')))
        h4 = lrelu(self.bn4(conv2d(h3, self.cf_dim*16, name='h4_conv', padding='VALID')))
        shape = np.product(h4.get_shape()[1:].as_list())
        h5 = tf.reshape(h4, [-1, shape])
        h6 = lrelu(linear(h5, 1024, 'd_h6_lin'))
        h7 = lrelu(linear(h6, 512, 'd_h7_lin'))
        logits = linear(h7, self.y_dim, 'd_co_lin')
        softmax = tf.nn.softmax(logits)

        return logits, softmax

    def train(self, dataset='wikiart', batch_size=64, epochs=25):
        tf.global_variables_initializer().run()
        self.dataset_name = dataset
        path = os.path.join('logs', 'classifier,dataset={},imsize={},batch_size={}').format(
            self.dataset_name, self.input_shape, batch_size)
        if not glob(path + "/*"):
            print(path + '/000')
            self.writer = SummaryWriter(path + '/000', self.sess.graph)
        else:
            nums = [int(x[-3]) for x in glob(path+"*")]
            num = str(max(nums) + 1)
            path = path + (3-len(num)) * "0" + num + '/'
            print(path)
            self.writer = SummaryWriter(path, self.sess.graph)

        self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))

        self.label_dict = {}
        path_list = glob('./data/', self.dataset_name + '/**/', recursive=True)[1:]
        for i, elem in enumerate(path_list):
            idx = len(self.dataset_name) + 8           
            print(elem[idx:-1])
            self.label_dict[elem[idx:-1]] = i
        shuffle(self.data)
        
        val_files = self.data[0:val_size]
        self.data = self.data[val_size:]
        val = [
          get_image(val_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for val_file in val_files]
        val_y = self.get_y(val_files)
        
        counter = 1
         
        for epoch in xrange(epochs):
            shuffle(self.data)
            batch_idxs = len(self.data) // batch_size
            
            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx*batch_size:(idx+1)*batch_size]
                batch = [
                    get_image(batch_file,
                        input_height=self.input_shape,
                        input_width=self.input_shape,
                        resize_height=self.input_shape,
                        resize_width=self.input_shape,
                        crop=False,
                        grayscale=False) for batch_file in batch_files]
                batch_labels = self.get_y(batch_files)
                
                loss, _, summary_str, acc = self.sess.run([self.loss, self.update, self.summary, self.accuracy], feed_dict={
                    self.inputs: batch_inputs,
                    self.y: batch_labels,
                })
                self.writer.add_summary(summary_str, counter)
                
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, acc: %.8f" \
                    % (epoch, idx, batch_idxs,
                    time.time() - start_time, errD, errG))
                if np.mod(counter, 500) == 1:
                    self.save(self.checkpoint_dir, counter)

        self.save(self.checkpoint_dir, counter)
    
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.input_shape, self.input_shape)
        
    def save(self, checkpoint_dir, step):
        model_name = "classifier.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)
    def get_y(self, sample_inputs):
        ret = []
        for sample in sample_inputs:
           _, _, _, lab_str, _ = sample.split('/', 4)
           ret.append(np.eye(self.y_dim)[np.array(self.label_dict[lab_str])])
        return ret

    


