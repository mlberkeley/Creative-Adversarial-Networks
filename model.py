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
from operator import mul
from slim.nets import nets_factory

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=32, smoothing=0.9, lamb = 1.0,
         use_resize=False, replay=True, style_net_checkpoint=None,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',wgan=False, can=True,
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, old_model=False):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.dataset_name = dataset_name
    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')
    self.d_bn5 = batch_norm(name='d_bn5')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.g_bn3 = batch_norm(name='g_bn3')
    self.g_bn4 = batch_norm(name='g_bn4')
    self.g_bn5 = batch_norm(name='g_bn5')

    # variables that determines whether to use style net separate from discriminator
    self.style_net_checkpoint = style_net_checkpoint

    self.smoothing = smoothing
    self.lamb = lamb

    self.can = can
    self.wgan = wgan
    self.use_resize = use_resize
    self.replay = replay

    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.experience_flag = False

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    elif self.dataset_name == 'wikiart':
      self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))

      self.c_dim = 3
      self.label_dict = {}
      path_list = glob('./data/wikiart/**/', recursive=True)[1:]
      for i, elem in enumerate(path_list):
        print(elem[15:-1])
        self.label_dict[elem[15:-1]] = i
    else:
      self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
      imreadImg = imread(self.data[0]);
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1
    self.experience_buffer=[]


    self.grayscale = (self.c_dim == 1)

    self.build_model(old_model=old_model)

  def upsample(self, input_, output_shape,
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
        name=None):
    if self.use_resize:
      return resizeconv(input_=input_, output_shape=output_shape,
        k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, name=(name or "resconv"))

    return deconv2d(input_=input_, output_shape=output_shape,
        k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, name= (name or "deconv2d"))
  def make_style_net(self, images):
    with tf.device("/gpu:0"):
      network_fn = nets_factory.get_network_fn(
          'inception_resnet_v2',
          num_classes=27,
          is_training=False)
      if images.shape[1:3] != (256, 256):
        # TODO check that the if statement works when the image shape is actually 256, 256
        images = tf.image.resize_images(images, [256, 256])
      logits, _ = network_fn(images)
      logits = tf.stop_gradient(logits)
      return logits

  def build_model(self, old_model=False):
    print('build_model', old_model)
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [None] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    if self.can:
      self.G                  = self.generator(self.z)
      self.D, self.D_logits, self.D_c, self.D_c_logits     = self.discriminator(
                                                                inputs, reuse=False, old_model=old_model)
      self.sampler            = self.sampler(self.z)





      if self.experience_flag:
        try:
          self.experience_selection = tf.convert_to_tensor(random.sample(self.experience_buffer, 16))
        except ValueError:
          self.experience_selection = tf.convert_to_tensor(self.experience_buffer)
        self.G = tf.concat([self.G, self.experience_selection], axis=0)

      self.D_, self.D_logits_, self.D_c_, self.D_c_logits_ = self.discriminator(
                                                                self.G, reuse=True, old_model=old_model)
      if self.style_net_checkpoint:
        self.style_net = self.make_style_net(self.G)
      self.d_sum = histogram_summary("d", self.D)
      self.d__sum = histogram_summary("d_", self.D_)
      self.d_c_sum = histogram_summary("d_c", self.D_c)
      self.d_c__sum = histogram_summary("d_c_", self.D_c_)
      self.G_sum = image_summary("G", self.G)

      correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.D_c,1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      true_label = tf.random_uniform(tf.shape(self.D),.8, 1.2)
      false_label = tf.random_uniform(tf.shape(self.D_), 0.0, 0.3)

      self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, true_label * tf.ones_like(self.D)))

      self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, false_label * tf.ones_like(self.D_)))

      self.d_loss_class_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=self.D_c_logits, labels=self.smoothing * self.y))
      if self.style_net_checkpoint:
        self.g_loss_class_fake = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits=self.style_net,
            labels=(1.0/self.y_dim)*tf.ones_like(self.style_net)))
      else:
        self.g_loss_class_fake = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits=self.D_c_logits,
            labels=(1.0/self.y_dim)*tf.ones_like(self.D_c_)))

      self.g_loss_fake = -tf.reduce_mean(tf.log(self.D_))

      self.d_loss = self.d_loss_real + self.d_loss_class_real + self.d_loss_fake
      self.g_loss = self.g_loss_fake + self.lamb * self.g_loss_class_fake

      self.d_loss_real_sum       = scalar_summary("d_loss_real", self.d_loss_real)
      self.d_loss_fake_sum       = scalar_summary("d_loss_fake", self.d_loss_fake)
      self.d_loss_class_real_sum = scalar_summary("d_loss_class_real", self.d_loss_class_real)
      self.g_loss_class_fake_sum = scalar_summary("g_loss_class_fake", self.g_loss_class_fake)


    else:
      self.G                  = self.generator(self.z, self.y)
      self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
      self.sampler            = self.sampler(self.z, self.y)
      self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

      self.d_sum = histogram_summary("d", self.D)
      self.d__sum = histogram_summary("d_", self.D_)
      self.G_sum = image_summary("G", self.G)

      self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, self.smoothing * tf.ones_like(self.D)))
      self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
      self.g_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

      self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
      self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

      self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    if self.style_net_checkpoint:
      all_vars = tf.trainable_variables()
      style_net_vars = [v for v in all_vars if 'InceptionResnetV2' in v.name]
      other_vars = [v for v in all_vars if 'InceptionResnetV2' not in v.name]
      self.saver = tf.train.Saver(var_list=other_vars)
      self.style_net_saver = tf.train.Saver(var_list=style_net_vars)
    else:
      self.saver=tf.train.Saver()
  def set_sess(self, sess):
    self.sess = sess
  def train(self, config):
    if self.sess is None:
      raise ValueError('Set session with set_sess(sess) before running')
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])

    if self.can:
      self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.d_loss_class_real_sum, self.g_loss_class_fake_sum])

    else:
      self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])


    # TODO refactor path = self.log_dir. Waiting for merge
    self.log_dir = config.log_dir

    self.writer = SummaryWriter(self.log_dir, self.sess.graph)

    #sample_z = n0random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    sample_z = np.random.normal(0, 1, size=[self.sample_num, self.z_dim])
    sample_z /= np.linalg.norm(sample_z, axis=0)

    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    elif self.y_dim:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
      sample_labels = self.get_y(sample_files)

    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir,
        config,
        style_net_checkpoint_dir=self.style_net_checkpoint)
    if could_load:
      counter = checkpoint_counter
      replay_files = glob(os.path.join(self.model_dir + '_replay'))
      self.experience_buffer =[
                    get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in replay_files]
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:
        #self.data = glob(os.path.join(
        # "./data", config.dataset, self.input_fname_pattern))
        shuffle(self.data)
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        self.experience_flag = not bool(idx % 2)

        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)
          batch_labels = self.get_y(batch_files)

        batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        batch_z /= np.linalg.norm(batch_z, axis=0)

        if self.can:
        #update D
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y: batch_labels,
            })
          self.writer.add_summary(summary_str,counter)
        #Update G: don't need labels or inputs
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z,
            })
          self.writer.add_summary(summary_str, counter)
          #do we need self.y for these two?
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z,
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z
          })

          errD_class_real = self.d_loss_class_real.eval({
              self.inputs: batch_images,
              self.y: batch_labels
          })
          errG_class_fake = self.g_loss_class_fake.eval({
              self.inputs: batch_images,
              self.z: batch_z
          })
          accuracy = self.accuracy.eval({
              self.inputs: batch_images,
              self.y: batch_labels
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z,
              self.y: batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z, self.y:batch_labels })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z,
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels
          })

        counter += 1
        if self.can:
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD_fake+errD_real+errD_class_real, errG))
          print("Discriminator class acc: %.2f" % (accuracy))
        else:
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD_fake+errD_real, errG))
        if np.mod(counter, 5) == 1:
          samp_images = self.G.eval({
              self.z: batch_z
          })
          if self.experience_flag:
            exp_path = os.path.join('buffer', self.model_dir)
            #max_ = get_max_end(exp_path)
            for i, image in enumerate(samp_images):
              #scipy.misc.imsave(exp_path + '_' + str(max_+i) + '.jpg', np.squeeze(image))
              self.experience_buffer.append(image)
            # todo make into a flag
            exp_buffer_len = 10000
            if len(self.experience_buffer) > exp_buffer_len:
              self.experience_buffer = self.experience_buffer[len(self.experience_buffer) - exp_buffer_len:]

        if np.mod(counter, 400) == 1:
          if config.dataset == 'mnist' or config.dataset == 'wikiart':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            except:
              print("one pic error!...")

        if np.mod(counter, config.save_itr) == 2:
          self.save(config.checkpoint_dir, counter, config)

  def discriminator(self, image, y=None, reuse=False, old_model=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      batch_dim = tf.shape(image)[0]
      if self.can:
        """
        256x256x3
        (4x4):
        32, 64, 128, 256, 512, 512
        doesn't use y, as it tries to predict y
        """
        #Common base of convolutions
        if old_model:
          h1 = lrelu(self.d_bn1(conv2d(image, self.df_dim*2, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        else:
          h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID'))
          h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, k_h=4, k_w=4, name='d_h3_conv', padding='VALID')))
        h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, k_h=4, k_w=4, name='d_h4_conv', padding='VALID')))
        shape = np.product(h4.get_shape()[1:].as_list())
        h5 = tf.reshape(h4, [-1, shape])
        #linear layer to determine if the image is real/fake
        r_out = linear(h5, 1, 'd_ro_lin')

        #fully connected layers to classify the image into the different styles.
        h6 = lrelu(linear(h5, 1024, 'd_h6_lin'))
        h7 = lrelu(linear(h6, 512, 'd_h7_lin'))
        c_out = linear(h7, self.y_dim, 'd_co_lin')
        c_softmax = tf.nn.softmax(c_out)

        return tf.nn.sigmoid(r_out), r_out, c_softmax, c_out
      else:
        yb = tf.reshape(y, [batch_dim, 1, 1, self.y_dim])
        image = conv_cond_concat(image, yb)
        h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, name='d_h0_conv',padding='VALID'))
        h0 = conv_cond_concat(h0, yb)
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*4, k_h=4, k_w=4, name='d_h1_conv', padding='VALID')))
        h1 = conv_cond_concat(h1, yb)
        #h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, name='d_h2_conv', padding='VALID')))
        #h2 = conv_cond_concat(h2, yb)
        #h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, k_h=4, k_w=4, name='d_h3_conv', padding='VALID')))
        #h3 = conv_cond_concat(h3, yb)
        #h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim*16, k_h=4, k_w=4, name='d_h4_conv', padding='VALID')))
        #h4 = conv_cond_concat(h4, yb)
        h5 = lrelu(self.d_bn5(conv2d(h1, self.df_dim*8, k_h=4, k_w=4, name='d_h5_conv', padding='VALID')))
        shape = np.product(h5.get_shape()[1:].as_list())
        h5 = tf.reshape(h5, [-1, shape])
        h5 = concat([h5,y],1)

        r_out = linear(tf.reshape(h5, [self.batch_size, -1]), 1, 'd_ro_lin')
        return tf.nn.sigmoid(r_out), r_out
  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if self.can:
        """
        for wikiart:
        R^100
        4 x 4 x 1024
        8 x 8 x 1024
        16 x 16 x 512
        32 x 32 x 256
        64 x 64 x 128
        128 x 128 x 64
        output: 256 x 256 x 3

        CAN does not use the y label to generate.

        self.gf_dim = 64

        """
        print(self.output_height, self.output_width)
        s_h, s_w = self.output_height, self.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)#8/8
        s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)#4/4

        z_ = linear(
            z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin')

        h0 = tf.reshape(
            z_, [-1, s_h64, s_w64, self.gf_dim * 16 ])
        h0 = lrelu(self.g_bn0(h0))

        h1 = self.upsample(
             h0, [-1, s_h32, s_w32, self.gf_dim*16], name='g_h1')
        h1 = lrelu(self.g_bn1(h1))

        h2 = self.upsample(
             h1, [-1, s_h16, s_w16, self.gf_dim*8], name='g_h2')
        h2 = lrelu(self.g_bn2(h2))

        h3 = self.upsample(
            h2, [-1, s_h8, s_w8, self.gf_dim*4], name='g_h3')
        h3 = lrelu(self.g_bn3(h3))

        h4 = self.upsample(
            h3, [-1, s_h4, s_w4, self.gf_dim*2], name='g_h4')
        h4 = lrelu(self.g_bn4(h4))

        h5 = self.upsample(
            h4, [-1, s_h2, s_w2, self.gf_dim], name='g_h5')
        h5 = lrelu(self.g_bn5(h5))

        h6 = self.upsample(
            h5, [-1, s_h, s_w, self.c_dim], name='g_h6')

        return tf.nn.tanh(h6)
      else:
        """
        this is just for GAN. only diff is the addition of conditioning vectors
        """
        s_h, s_w = self.output_height, self.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        #s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16
        #s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)#8/8
        #s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)#4/4

        # project `z` and reshape

        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z,y],1)
        #z_ = linear(
        #    z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin')
        z_ = linear(
             z, 4 * self.gf_dim * s_h8 * s_w8, 'g_h0_lin')
        #h0 = tf.reshape(
        #    z_, [-1, s_h64, s_w64, self.gf_dim * 16])
        #h0 = lrelu(self.g_bn0(h0))
        #h0 = conv_cond_concat(h0, yb)

        #h1 = self.upsample(
        #     h0, [-1, s_h32, s_w32, self.gf_dim*16], name='g_h1')
        #h1 = lrelu(self.g_bn1(h1))
        #h1 = conv_cond_concat(h1, yb)

        #h2 = self.upsample(
        #     h1, [-1, s_h16, s_w16, self.gf_dim*8], name='g_h2')
        #h2 = lrelu(self.g_bn2(h2))
        #h2 = conv_cond_concat(h2, yb)
        h2 = tf.reshape(
            z_, [-1, s_h8, s_w8, self.gf_dim*4])
        h3 = self.upsample(
            h2, [-1, s_h8, s_w8, self.gf_dim*4], name='g_h3')
        h3 = lrelu(self.g_bn3(h3))
        h3 = conv_cond_concat(h3, yb)

        h4 = self.upsample(
            h3, [-1, s_h4, s_w4, self.gf_dim*2], name='g_h4')
        h4 = lrelu(self.g_bn4(h4))
        h4 = conv_cond_concat(h4, yb)

        h5 = self.upsample(
            h4, [-1, s_h2, s_w2, self.gf_dim], name='g_h5')
        h5 = lrelu(self.g_bn5(h5))
        h5 = conv_cond_concat(h5, yb)

        h6 = self.upsample(
            h5, [-1, s_h, s_w, self.c_dim], name='g_h6')
        return tf.nn.tanh(h6)
  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      if self.can:
        s_h, s_w = self.output_height, self.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)#8/8
        s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)#4/4

        # project `z` and reshape
        z_ = linear(z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin')

        h0 = tf.reshape(z_, [-1, s_h64, s_w64, self.gf_dim * 16])
        h0 = lrelu(self.g_bn0(h0, train=False))


        h1 = self.upsample(h0, [-1, s_h32, s_w32, self.gf_dim*16], name='g_h1')
        h1 = lrelu(self.g_bn1(h1, train=False))

        h2 = self.upsample(h1, [-1, s_h16, s_w16, self.gf_dim*8], name='g_h2')
        h2 = lrelu(self.g_bn2(h2, train=False))

        h3 = self.upsample(h2, [-1, s_h8, s_w8, self.gf_dim*4], name='g_h3')
        h3 = lrelu(self.g_bn3(h3, train=False))

        h4 = self.upsample(h3, [-1, s_h4, s_w4, self.gf_dim*2], name='g_h4')
        h4 = lrelu(self.g_bn4(h4, train=False))

        h5 = self.upsample(h4, [-1, s_h2, s_w2, self.gf_dim], name='g_h5')
        h5 = lrelu(self.g_bn5(h5, train=False))

        h6 = self.upsample(h5, [-1, s_h, s_w, self.c_dim], name='g_h6')

        return tf.nn.tanh(h6)
      else:
        s_h, s_w = self.output_height, self.output_width #256/256
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)      #128/128
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)    #64/64
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)    #32/32
        #s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  #16/16
        #s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)#8/8
        #s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)#4/4

        yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        z = concat([z,y],1)
        #z_ = linear(z, self.gf_dim*16*s_h64*s_w64, 'g_h0_lin')
        z_ = linear(z, 4 * self.gf_dim * s_h8 * s_w8, 'g_h0_lin')
        h2 = tf.reshape(z_, [-1, s_h8, s_w8, self.gf_dim * 4])
        #h0 = tf.reshape(z_, [-1, s_h64, s_w64, self.gf_dim * 16])
        #h0 = lrelu(self.g_bn0(h0, train=False))
        #h0 = conv_cond_concat(h0,yb)

        #Unlike the original paper, we use resize convolutions to avoid checkerboard artifacts.

        #h1 = self.upsample(h0, [-1, s_h32, s_w32, self.gf_dim*16], name='g_h1')
        #h1 = lrelu(self.g_bn1(h1, train=False))
        #h1 = conv_cond_concat(h1,yb)

        #h2 = self.upsample(h1, [-1, s_h16, s_w16, self.gf_dim*8], name='g_h2')
        #h2 = lrelu(self.g_bn2(h2, train=False))
        #h2 = conv_cond_concat(h2,yb)

        h3 = self.upsample(h2, [-1, s_h8, s_w8, self.gf_dim*4], name='g_h3')
        h3 = lrelu(self.g_bn3(h3, train=False))
        h3 = conv_cond_concat(h3,yb)

        h4 = self.upsample(h3, [-1, s_h4, s_w4, self.gf_dim*2], name='g_h4')
        h4 = lrelu(self.g_bn4(h4, train=False))
        h4 = conv_cond_concat(h4,yb)

        h5 = self.upsample(h4, [-1, s_h2, s_w2, self.gf_dim], name='g_h5')
        h5 = lrelu(self.g_bn5(h5, train=False))
        h5 = conv_cond_concat(h5,yb)

        h6 = self.upsample(h5, [-1, s_h, s_w, self.c_dim], name='g_h6')

        return tf.nn.tanh(h6)

  def get_y(self, sample_inputs):
    ret = []
    for sample in sample_inputs:
      _, _, _, lab_str, _ = sample.split('/', 4)
      ret.append(np.eye(self.y_dim)[np.array(self.label_dict[lab_str])])
    return ret

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)

    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step, config):
    model_name = "DCGAN.model"
    if not config.use_default_checkpoint:
      checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

    if config.use_s3:
      import aws
      s3_dir = checkpoint_dir
      aws.upload_path(checkpoint_dir, config.s3_bucket, s3_dir)
      print('uploading log')
      aws.upload_path(self.log_dir, config.s3_bucket, self.log_dir, certain_upload=True)


  def load_specific(self, checkpoint_dir):
    ''' like loading but takes in a directory directly'''
    import re
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  def load(self, checkpoint_dir, config, style_net_checkpoint_dir=None):
    import re
    print(" [*] Reading checkpoints...")
    if not config.use_default_checkpoint:
      checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if style_net_checkpoint_dir is not None:
      ckpt = tf.train.get_checkpoint_state(style_net_checkpoint_dir)
      if not ckpt:
        raise ValueError('style_net_checkpoint_dir points to wrong directory/model doesn\'t exist')
      ckpt_name = os.path.join(style_net_checkpoint_dir, os.path.basename(ckpt.model_checkpoint_path))
      self.style_net_saver.restore(self.sess, tf.train.latest_checkpoint(style_net_checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0


