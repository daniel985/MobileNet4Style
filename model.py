import tensorflow as tf

def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv') as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        #weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(shape))
        #x_padded = tf.pad(x, [[0, 0], [kernel / 2, kernel / 2], [kernel / 2, kernel / 2], [0, 0]], mode=mode)
        return tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name='conv')

def conv2d_dw(x, input_filters, multiplier, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv_dw') as scope:
        shape = [kernel, kernel, input_filters, multiplier]
        #weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(shape))
        #x_padded = tf.pad(x, [[0, 0], [kernel / 2, kernel / 2], [kernel / 2, kernel / 2], [0, 0]], mode=mode)
        conv_dw = tf.nn.depthwise_conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name='conv_dw')
        return conv_dw

def conv2d_transpose(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose') as scope:

        shape = [kernel, kernel, output_filters, input_filters]
        #weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(shape))

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], name='conv_transpose')


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    '''
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument
    '''
    with tf.variable_scope('conv_transpose') as scope:
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return conv2d(x_resized, input_filters, output_filters, kernel, strides)

def resize(x, strides, training):
    with tf.variable_scope('conv_transpose') as scope:
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2
        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x_resized

def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def batch_norm(x, size, training, decay=0.999):
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    pop_mean = tf.Variable(tf.zeros([size]))
    pop_var = tf.Variable(tf.ones([size]))
    epsilon = 1e-3

    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch_norm')

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon, name='batch_norm')

    return tf.cond(training, batch_statistics, population_statistics)


def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual') as scope:
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides)

        residual = x + conv2

        return residual

def mresidual(x, filters, kernel, strides):
    with tf.variable_scope('mresidual') as scope:
        conv1_dw = conv2d_dw(x, filters, 1, kernel, strides)
        conv1 = conv2d(conv1_dw, filters, filters, 1, 1)

        #conv2_dw = conv2d_dw(tf.nn.relu(conv1), filters, 1, kernel, strides)
        #conv2 = conv2d(conv2_dw, filters, filters, 1, 1)
        return x + conv1

def net(image, alpha):
    with tf.variable_scope('conv1'):
        mnet = tf.nn.relu(instance_norm(conv2d(image, 3, int(32*alpha), 3, 1)))

    with tf.variable_scope('conv2_dw'):
        mnet = tf.nn.relu(instance_norm(conv2d_dw(mnet, int(32*alpha), 1, 3, 2)))
    with tf.variable_scope('conv2'):
        mnet = tf.nn.relu(instance_norm(conv2d(mnet, int(32*alpha), int(64*alpha), 1, 1)))
    with tf.variable_scope('conv3_dw'):
        mnet = tf.nn.relu(instance_norm(conv2d_dw(mnet, int(64*alpha), 1, 3, 2)))
    with tf.variable_scope('conv3'):
        mnet = tf.nn.relu(instance_norm(conv2d(mnet, int(64*alpha), int(128*alpha), 1, 1)))

    #with tf.variable_scope('res1'):
    #    mnet = mresidual(mnet, int(128*alpha), 3, 1)
    #with tf.variable_scope('res2'):
    #    mnet = mresidual(mnet, int(128*alpha), 3, 1)
    #with tf.variable_scope('res3'):
    #    mnet = mresidual(mnet, int(128*alpha), 3, 1)
    #with tf.variable_scope('res4'):
    #    mnet = mresidual(mnet, int(128*alpha), 3, 1)
    #with tf.variable_scope('res5'):
    #    mnet = mresidual(mnet, int(128*alpha), 3, 1)
    
    with tf.variable_scope('conv4_dw'):
        mnet = tf.nn.relu(instance_norm(conv2d_dw(mnet, int(128*alpha), 1, 3, 1)))
    with tf.variable_scope('conv4'):
        mnet = tf.nn.relu(instance_norm(conv2d(mnet, int(128*alpha), int(128*alpha), 1, 1)))
    with tf.variable_scope('conv5_dw'):
        mnet = tf.nn.relu(instance_norm(conv2d_dw(mnet, int(128*alpha), 1, 3, 1)))
    with tf.variable_scope('conv5'):
        mnet = tf.nn.relu(instance_norm(conv2d(mnet, int(128*alpha), int(128*alpha), 1, 1)))
    with tf.variable_scope('conv6_dw'):
        mnet = tf.nn.relu(instance_norm(conv2d_dw(mnet, int(128*alpha), 1, 3, 1)))
    with tf.variable_scope('conv6'):
        mnet = tf.nn.relu(instance_norm(conv2d(mnet, int(128*alpha), int(128*alpha), 1, 1)))

    with tf.variable_scope('deconv1_dw'):
        mnet = resize(mnet, 2, training)
        mnet = tf.nn.relu(instance_norm(conv2d_dw(mnet, int(128*alpha), 1, 3, 2)))
    with tf.variable_scope('deconv1'):
        mnet = tf.nn.relu(instance_norm(conv2d(mnet, int(128*alpha), int(64*alpha), 1, 1)))
    with tf.variable_scope('deconv2_dw'):
        mnet = resize(mnet, 2, training)
        mnet = tf.nn.relu(instance_norm(conv2d_dw(mnet, int(64*alpha), 1, 3, 2)))
    with tf.variable_scope('deconv2'):
        mnet = tf.nn.relu(instance_norm(conv2d(mnet, int(64*alpha), int(32*alpha), 1, 1)))

    with tf.variable_scope('final'):
        mnet = tf.nn.tanh(instance_norm(conv2d(mnet, int(32*alpha), 3, 3, 1)))

    y = (mnet + 1) * 127.5

    return y
