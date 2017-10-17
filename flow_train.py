# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from nets import nets_factory
from preprocessing import preprocessing_factory
from skimage import io, color
import flow_reader
import model
import time
import flow_losses
import utils
import os
import cv2
import argparse

slim = tf.contrib.slim

tf.app.flags.DEFINE_string("naming", "train", "Name of this run")
tf.app.flags.DEFINE_string("style", "", "Name of this style")
tf.app.flags.DEFINE_string("data_path", "/data/train_video/256x256p/cows", "")
tf.app.flags.DEFINE_float("content_weight", 1.0, "Weight for content features loss")
tf.app.flags.DEFINE_float("style_weight", 1.0, "Weight for style features loss")
tf.app.flags.DEFINE_float("tv_weight", 0.0, "Weight for total variation loss")
tf.app.flags.DEFINE_float("flow_weight", 1.0, "Weight for flow loss")
tf.app.flags.DEFINE_float("alpha", 1.0, "Control model filters")
tf.app.flags.DEFINE_string("loss_model", "vgg_16","Path to vgg model weights")
tf.app.flags.DEFINE_string("loss_model_file", "pretrained/vgg_16.ckpt","Path to vgg model weights")
tf.app.flags.DEFINE_string("checkpoint_exclude_scopes", "vgg_16/fc","")
tf.app.flags.DEFINE_string("model_path", "models","")
tf.app.flags.DEFINE_string("content_layers", "vgg_16/conv3/conv3_3","Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("style_layers", "vgg_16/conv1/conv1_2,vgg_16/conv2/conv2_2,vgg_16/conv3/conv3_3,vgg_16/conv4/conv4_3","Which layers to extract style from")
tf.app.flags.DEFINE_string("style_layers_weights", "0.2,0.2,0.2,0.2","Each layers weight")
tf.app.flags.DEFINE_string("style_image", "style.png", "Styles to train")
tf.app.flags.DEFINE_integer("image_size", 256, "Size of output image")
tf.app.flags.DEFINE_integer("style_size", 1024, "")
tf.app.flags.DEFINE_integer("batch_size", 2, "Number of concurrent images to train on")
tf.app.flags.DEFINE_integer("epoch", 2, "Number of concurrent images to train on")
tf.app.flags.DEFINE_string("device", "0", "device")
tf.app.flags.DEFINE_integer("max_iter", 1e6, "max_iter")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device
MEAN_VALUES = np.array([123.68, 116.78, 103.94])

def main(argv=None):
    content_layers = FLAGS.content_layers.split(',')
    style_layers = FLAGS.style_layers.split(',')
    style_layers_weights = [float(i) for i in FLAGS.style_layers_weights.split(",")]
    #num_steps_decay = 82786 * 2 / FLAGS.batch_size
    
    dirnames = []
    for root, dirs, files in os.walk(FLAGS.data_path):
        dirnames += [[os.path.join(root, name) for name in files]]

    style_features_t = flow_losses.get_style_features(FLAGS)
    training_path = os.path.join(FLAGS.model_path, FLAGS.naming)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)

    with tf.Session() as sess:
        """Build Network"""
        network_fn = nets_factory.get_network_fn(FLAGS.loss_model,num_classes=1,is_training=False)
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.loss_model,is_training=False)
        
        image_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
        flow_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size - 1, FLAGS.image_size, FLAGS.image_size, 2])

        generated = model.net(image_placeholder, FLAGS.alpha)
        processed_generated = [image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)
                for image in tf.unstack(generated, axis=0, num=FLAGS.batch_size)]
        processed_generated = tf.stack(processed_generated)
        _, endpoints_dict = network_fn(tf.concat([processed_generated, image_placeholder], 0), spatial_squeeze=False)

        """Build Losses"""
        content_loss = flow_losses.content_loss(endpoints_dict, content_layers)
        style_loss, style_losses = flow_losses.style_loss(endpoints_dict, style_features_t, style_layers, style_layers_weights)
        tv_loss = flow_losses.total_variation_loss(generated)  # use the unprocessed image
        flow_loss = flow_losses.flow_loss(generated, flow_placeholder, FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size)

        content_loss = FLAGS.content_weight * content_loss
        style_loss = FLAGS.style_weight * style_loss
        tv_loss = FLAGS.tv_weight * tv_loss
        flow_loss = FLAGS.flow_weight * flow_loss
        loss = style_loss + content_loss + tv_loss + flow_loss

        """Prepare to Train"""
        global_step = tf.Variable(0, name="global_step", trainable=False)
        variable_to_train = []
        for variable in tf.trainable_variables():
            if not(variable.name.startswith(FLAGS.loss_model)):
                variable_to_train.append(variable)
        
        lr = tf.train.exponential_decay(
                learning_rate = 1e-3,
                global_step = global_step,
                decay_steps = 100000,
                decay_rate = 1e-1,
                staircase = True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-8)
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=variable_to_train)
        #train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)
        variables_to_restore = []
        for v in tf.global_variables():
            if not(v.name.startswith(FLAGS.loss_model)):
                variables_to_restore.append(v)
        saver = tf.train.Saver(variables_to_restore)
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        init_func = utils._get_init_fn(FLAGS)
        init_func(sess)
        last_file = tf.train.latest_checkpoint(training_path)
        if last_file:
            print ('Restoring model from {}'.format(last_file))
            saver.restore(sess, last_file)

        """Start Training"""
        for filenames in dirnames:
            images = []
            images_gray = []
            for filename in sorted(filenames):
                print(filename)
                image_bytes = io.imread(filename)
                image = image_bytes - MEAN_VALUES
                image_gray = color.rgb2grey(image_bytes)
                images.append(image_bytes)
                images_gray.append(image_gray)
            flows = []
            for i in xrange(len(images_gray) - 1):
                flow = cv2.calcOpticalFlowFarneback(images_gray[i], images_gray[i+1], 0.5, 3, 15, 3, 5, 1.2, 0) 
                flows.append(flow)

            for i in xrange(len(images) - FLAGS.batch_size):
                feed_dict = {
                        image_placeholder : np.stack(images[i:i+FLAGS.batch_size]),
                        flow_placeholder : np.stack(flows[i:i+FLAGS.batch_size-1])
                        }

                _, c_loss, s_losses, t_loss, f_loss, total_loss, step = sess.run([train_op, content_loss, style_losses, tv_loss, flow_loss, loss, global_step],
                        feed_dict=feed_dict)

                """logging"""
                if step % 10 == 0:
                    print(step, c_loss, s_losses, t_loss, f_loss, total_loss)
                """checkpoint"""
                if step % 10000 == 0:
                    saver.save(sess, os.path.join(training_path, 'flow-loss-model'), global_step=step)
                if step == FLAGS.max_iter:
                    saver.save(sess, os.path.join(training_path, 'flow-loss-model-done'))
                    print ("Save flow-loss-model done!")
                    return
        if step < FLAGS.max_iter:
            saver.save(sess, os.path.join(training_path, 'flow-loss-model-done'))
            print ("Save flow-loss-model done")

if __name__ == '__main__':
    tf.app.run()
