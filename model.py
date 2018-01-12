from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from random import shuffle

from slim.nets import nets_factory
import generators
import discriminators

from ops import *
from utils import *
from losses import *

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=32, smoothing=0.9, lamb = 1.0,

         use_resize=False, replay=False, learning_rate = 1e-4, style_net_checkpoint=None,
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
    self.learning_rate = learning_rate


    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn0 = batch_norm(name='d_bn0')
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
        images = tf.image.resize_images(images, [256, 256])
      logits, _ = network_fn(images)
      logits = tf.stop_gradient(logits)
      return logits
  def set_sess(self, sess):
    ''' set session to sess '''
    self.sess = sess

  def build_model(self, old_model=False):
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
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    if self.wgan and not self.can:
        self.discriminator = discriminators.dcwgan_cond
        self.generator = generators.dcgan_cond
        self.d_update, self.g_update, self.losses, self.sums = WGAN_loss(self)

    if self.wgan and self.can:
        self.discriminator = discriminators.vanilla_wgan
        self.generator = generators.vanilla_wgan
        #TODO: write all this wcan stuff
        self.d_update, self.g_update, self.losses, self.sums = WCAN_loss(self)
    if not self.wgan and self.can:
        self.discriminator = discriminators.vanilla_can
        self.generator = generators.vanilla_can
        self.d_update, self.g_update, self.losses, self.sums = CAN_loss(self)
    elif not self.wgan and not self.can:
        #TODO: write the regular gan stuff
        self.d_update, self.g_update, self.losses, self.sums = GAN_loss(self)

    if self.can or not self.y_dim:
        self.sampler            = self.generator(self, self.z, is_sampler=True)
    else:
        self.sampler            = self.generator(self, self.z, self.y, is_sampler=True)

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
  def train(self, config):
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()


    self.log_dir = config.log_dir

    self.writer = SummaryWriter(self.log_dir, self.sess.graph)


    sample_z = np.random.normal(0, 1, [self.sample_num, self.z_dim]) \
              .astype(np.float32)
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
    could_load, checkpoint_counter, loaded_sample_z = self.load(self.checkpoint_dir,
        config,
        style_net_checkpoint_dir=self.style_net_checkpoint)
    if could_load:
      counter = checkpoint_counter
      if self.replay:
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
      if loaded_sample_z is not None:
        sample_z = loaded_sample_z
    else:
      print(" [!] Load failed...")

    np.save(os.path.join(self.checkpoint_dir, 'sample_z'), sample_z)
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

          _, summary_str = self.sess.run([self.d_update, self.sums[0]],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y: batch_labels,
            })
          self.writer.add_summary(summary_str,counter)
        #Update G: don't need labels or inputs
          _, summary_str = self.sess.run([self.g_update, self.sums[1]],
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
          if self.wgan:
            for i in range(4):
              _, summary_str = self.sess.run([self.d_update, self.d_sum],
                feed_dict={
                  self.inputs: batch_images,
                  self.z: batch_z,
                  self.y: batch_labels,
              })
              self.writer.add_summary(summary_str, counter)
              slopes = self.sess.run(self.slopes,

                feed_dict={
                  self.inputs: batch_images,
                  self.z: batch_z,
                  self.y: batch_labels

              })
          _, summary_str = self.sess.run([self.d_update, self.d_sum],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([self.g_update, self.g_sum],
            feed_dict={
              self.z: batch_z,
              self.y: batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          errD = self.d_loss.eval({
              self.inputs: batch_images,
              self.y:batch_labels,
              self.z:batch_z
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
          if self.wgan:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD, errG))
          else:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, idx, batch_idxs,
              time.time() - start_time, errD, errG))

        if np.mod(counter, 5) == 1 and self.replay:
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


        if np.mod(counter, config.sample_itr) == 1:

          if config.dataset == 'mnist' or config.dataset == 'wikiart':
            samples = self.sess.run(
              self.sampler,
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
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

  def load(self, checkpoint_dir, config, style_net_checkpoint_dir=None, use_last_checkpoint=True):
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

    # finds teh checkpoint
    if config.use_default_checkpoint and use_last_checkpoint:
      def get_parent_path(path):
        return os.path.normpath(os.path.join(path, os.pardir))
      path = get_parent_path(get_parent_path( checkpoint_dir))
      #find the high checkpoint path in a path
      files_in_path = sorted(os.listdir(path))

      if len(files_in_path) > 1:
        last_ = files_in_path[-2]

        checkpoint_dir  = os.path.join(path, last_, 'checkpoint')
      else:
        checkpoint = None


    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      if os.path.exists(os.path.join(checkpoint_dir, 'sample_z.npy')):
        print(" [*] Success to read sample_z in {}".format(ckpt_name))
        sample_z = np.load(os.path.join(checkpoint_dir, 'sample_z.npy'))
      else:
        print(" [*] Failed to find a sample_z")
        sample_z = None
      return True, counter, sample_z
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0, None


