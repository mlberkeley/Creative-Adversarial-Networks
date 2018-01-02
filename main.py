import os
import scipy.misc
import numpy as np
from glob import glob

from model import DCGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("smoothing", 0.9, "Smoothing term for discriminator real (class) loss [0.9]")
flags.DEFINE_float("lambda_val", 1.0, "determines the relative importance of style ambiguity loss [1.0]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("save_itr", 500, "The number of iterations to run for saving checkpoints")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 64, "the size of sample images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("log_dir", 'logs', "Directory to store logs [logs]")
flags.DEFINE_string("checkpoint_dir", None, "Directory name to save the checkpoints [<FLAGS.log_dir>/checkpoint]")
flags.DEFINE_string("sample_dir", None, "Directory name to save the image samples [<FLAGS.log_dir>/samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("wgan", False, "True if WGAN, False if regular [G/C]AN [False]")
flags.DEFINE_boolean("can", True, "True if CAN, False if regular GAN [True]")
flags.DEFINE_boolean("use_s3", False, "True if you want to use s3 buckets, False if you don't. Need to set s3_bucket if True.")
flags.DEFINE_string("s3_bucket", None, "the s3_bucket to upload results to")
flags.DEFINE_boolean("replay", True, "True if using experience replay [True]")
flags.DEFINE_boolean("use_resize", False, "True if resize conv for upsampling, False for fractionally strided conv [False]")
flags.DEFINE_boolean("use_default_checkpoint", False, "True only if checkpoint_dir is None. Don't set this")
flags.DEFINE_string("style_net_checkpoint", None, "The checkpoint to get style net. Leave default to note use stylenet")
FLAGS = flags.FLAGS

def main(_):
  print('Before processing flags')
  pp.pprint(flags.FLAGS.__flags)
  if FLAGS.use_s3:
    import aws
    if FLAGS.s3_bucket is None:
      raise ValueError('use_s3 flag set, but no bucket set. ')
    # check to see if s3 bucket exists:
    elif not aws.bucket_exists(FLAGS.s3_bucket):
      raise ValueError('`use_s3` flag set, but bucket "%s" doesn\'t exist. Not using s3' % FLAGS.s3_bucket)


  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height



  # configure the log_dir to match the params
  log_dir = os.path.join(FLAGS.log_dir, "dataset={},isCan={},lr={},imsize={},batch_size={}".format(
                FLAGS.dataset,
                FLAGS.can,
                FLAGS.learning_rate,
                FLAGS.input_height,
                FLAGS.batch_size))
  if not glob(log_dir + "*"):
    log_dir = os.path.join(log_dir, "000")
  else:
    containing_dir=os.path.join(log_dir, "*")
    nums = [int(x[-3:]) for x in glob(containing_dir)] # TODO FIX THESE HACKS
    if nums == []:
      num = 0
    else:
      num = max(nums) + 1
    log_dir = os.path.join(log_dir,"{:03d}".format(num))
  FLAGS.log_dir = log_dir

  if FLAGS.checkpoint_dir is None:
    FLAGS.checkpoint_dir = os.path.join(FLAGS.log_dir, 'checkpoint')
    FLAGS.use_default_checkpoint = True
  elif FLAGS.use_default_checkpoint:
    raise ValueError('`use_default_checkpoint` flag only works if you keep checkpoint_dir as None')

  if FLAGS.sample_dir is None:
    FLAGS.sample_dir = os.path.join(FLAGS.log_dir, 'samples')

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  print('After processing flags')
  pp.pprint(flags.FLAGS.__flags)
  if FLAGS.style_net_checkpoint:
    from slim.nets import nets_factory
    network_fn = nets_factory


  sess = None
  if FLAGS.dataset == 'mnist':
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.sample_size,
        use_resize=FLAGS.use_resize,
        replay=FLAGS.replay,
        y_dim=10,
        smoothing=FLAGS.smoothing,
        lamb = FLAGS.lambda_val,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        wgan=FLAGS.wgan,
        can=FLAGS.can)
  elif FLAGS.dataset == 'wikiart':
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.sample_size,
        use_resize=FLAGS.use_resize,
        replay=FLAGS.replay,
        y_dim=27,
        smoothing=FLAGS.smoothing,
        lamb = FLAGS.lambda_val,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        wgan=FLAGS.wgan,
        style_net_checkpoint=FLAGS.style_net_checkpoint,
        can=FLAGS.can)
  else:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.sample_size,
        dataset_name=FLAGS.dataset,
        replay=FLAGS.replay,
        input_fname_pattern=FLAGS.input_fname_pattern,
        use_resize=FLAGS.use_resize,
        smoothing=FLAGS.smoothing,
        crop=FLAGS.crop,
        lamb = FLAGS.lambda_val,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        wgan=FLAGS.wgan,
        can=FLAGS.can)

  # run_config = tf.ConfigProto(log_device_placement=True)
  run_config = tf.ConfigProto()
  # run_config.gpu_options.allow_growth=True
  with tf.Session(config=run_config) as sess:
    dcgan.set_sess(sess)
    # show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    OPTION = 0
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
