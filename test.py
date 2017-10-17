# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
from nets import nets_factory
from preprocessing import preprocessing_factory
import scipy.misc
import reader
import model
import time
import os

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_string("model_file", "models/canvas2/fast-style-model-done", "")
tf.app.flags.DEFINE_float("alpha", 1.0, "Control model filters")
tf.app.flags.DEFINE_string("image_file", "org.jpg", "")
tf.app.flags.DEFINE_string("save_file", "mnet.jpg", "")
tf.app.flags.DEFINE_string("device", "0", "")

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device

def main(_):
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(FLAGS.loss_model,is_training=False)
            image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
            image = tf.expand_dims(image, 0)
            generated = model.net(image, FLAGS.alpha)
            generated = tf.squeeze(generated, [0])
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            start_time = time.time()
            generated = sess.run(generated)
            end_time = time.time()
            print ("cost time: %f"%(end_time - start_time))

            generated = tf.cast(generated, tf.uint8).eval()
            scipy.misc.imsave(FLAGS.save_file, generated)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
