# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import utils
import os
import cv2

slim = tf.contrib.slim


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


def get_style_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """

    style_layers = FLAGS.style_layers.split(',')
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False)

        size = FLAGS.style_size
        img_bytes = tf.read_file(FLAGS.style_image)
        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)
        # image = _aspect_preserving_resize(image, size)
        images = tf.stack([image_preprocessing_fn(image, size, size)])
        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        for layer in style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            features.append(feature)

        with tf.Session() as sess:
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)
            return sess.run(features)


def style_loss(endpoints_dict, style_features_t, style_layers, style_layers_weights):
    style_loss = 0
    style_loss_summary = []
    for style_gram, layer, weight in zip(style_features_t, style_layers, style_layers_weights):
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        #generated_gram = gram(generated_images)
        #size = tf.size(generated_gram)
        #layer_style_loss = tf.nn.l2_loss(generated_gram - style_gram) * 2 / tf.to_float(size)
        style_loss += weight * layer_style_loss
        style_loss_summary.append(weight * layer_style_loss)
    return style_loss, style_loss_summary


def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  # remain the same as in the paper
        #content_loss += tf.nn.l2_loss(generated_images - content_images) / ((h*w)**0.5 * (d)**0.5 * b)
    return content_loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss

def frame_loss(Y, batch_size):
    frame_loss = 0
    b,h,w,d = tf.unstack(tf.shape(Y))
    for i in xrange(batch_size - 1):
        frame_loss += tf.nn.l2_loss(Y[i+1,:,:,:] - Y[i,:,:,:]) * 2 / tf.to_float(h*w*d)
    return frame_loss / (batch_size-1)

def flow_loss(Y, flow_X, batch_size, height, width):
    flow_loss = 0
    b,h,w,d = tf.unstack(tf.shape(Y))
    for i in xrange(batch_size - 1):
        flow_Y = warp_image(Y[i,:,:,:], flow_X[i], height, width)
        #flow_loss += tf.nn.l2_loss(Y[i+1,:,:,:] - flow_Y) * 2 / tf.to_float(h*w*d)
        flow_loss += tf.nn.l2_loss(flow_Y - Y[i,:,:,:]) * 2 / tf.to_float(h*w*d)
    return flow_loss / (batch_size-1)

def warp_image(tf_image, tf_flow, width, height):
    weight_flows = tf_flow - tf.floor(tf_flow)
    floor_flows = tf.to_int32(tf.floor(tf_flow))
    floor_flat = tf.reshape(floor_flows, [-1, 2])
    floor_flows = floor_flat
    image_flat = tf.reshape(tf_image, [-1, 3])
    weight_flat = tf.reshape(weight_flows, [-1, 2])
    weight_flows = weight_flat
    x = floor_flows[:,0]
    y = floor_flows[:,1]
    xw = weight_flows[:,0]
    yw = weight_flows[:,1]
    pos_x = tf.range(height)
    pos_x = tf.tile(tf.expand_dims(pos_x, 1), [1, width])
    pos_x = tf.reshape(pos_x, [-1])
    pos_y = tf.range(width)
    pos_y = tf.tile(tf.expand_dims(pos_y, 0), [height, 1])
    pos_y = tf.reshape(pos_y, [-1])
    zero = tf.zeros([], dtype='int32')

    channels = []
    for c in range(3):
        x0 = pos_y + x
        x1 = x0 + 1
        y0 = pos_x + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, width-1)
        x1 = tf.clip_by_value(x1, zero, width-1)
        y0 = tf.clip_by_value(y0, zero, height-1)
        y1 = tf.clip_by_value(y1, zero, height-1)

        idx_a = y0 * width + x0
        idx_b = y1 * width + x0
        idx_c = y0 * width + x1
        idx_d = y1 * width + x1

        Ia = tf.gather(image_flat[:, c], idx_a)
        Ib = tf.gather(image_flat[:, c], idx_b)
        Ic = tf.gather(image_flat[:, c], idx_c)
        Id = tf.gather(image_flat[:, c], idx_d)

        wa = (1-xw) * (1-yw)
        wb = (1-xw) * yw
        wc = xw * (1-yw)
        wd = xw * yw

        img = tf.multiply(Ia, wa) + tf.multiply(Ib, wb) + tf.multiply(Ic, wc) + tf.multiply(Id, wd)
        channels.append(tf.reshape(img, shape=(height, width)))
    return tf.stack(channels, axis=-1)
