# ============================================================== #
#                            Layers                              #
#                                                                #
#                                                                #
# Higher level operations for quickly building layers            #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers.python.layers import layers as tf_layers

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import numpy as np


def conv(inputs, kernel_size, num_outputs, name,
        stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution layer followed by activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv_btn(inputs, kernel_size, num_outputs, name,
        is_training = True, stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution layer followed by batch normalization then activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        is_training: Boolean, in training mode or not
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value
        kernel_shape   = [kernel_size[0], kernel_size[1], num_filters_in, num_outputs]
        stride_shape   = [1, stride_size[0], stride_size[1], 1]

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv    = tf.nn.conv2d(inputs, weights, stride_shape, padding = padding)
        outputs = tf.nn.bias_add(conv, bias)
        outputs = tf.contrib.layers.batch_norm(outputs, center = True, scale = True, is_training = is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def deconv(inputs, kernel_size, num_filters_in, num_outputs, name,
        stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution Transpose followed by activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_filters_in: Ingteger, number of channels in input tensor
        num_outputs: Integer, number of convolution filters
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], num_outputs, num_filters_in]
        stride_shape = [1, stride_size[0], stride_size[1], 1]
        input_shape  = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], input_shape[1], input_shape[2], num_outputs])

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv_trans = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)
        outputs    = tf.nn.bias_add(conv_trans, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def deconv_btn(inputs, kernel_size, num_filters_in, num_outputs, name,
        is_training = True, stride_size = [1, 1], padding = 'SAME', activation_fn = tf.nn.relu):
    """
    Convolution Transpose followed by batch normalization then activation fn:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        num_filters_in: Ingteger, number of channels in input tensor
        num_outputs: Integer, number of convolution filters
        is_training: Boolean, in training mode or not
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height+-, width+-, num_outputs]
    """

    with tf.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], num_outputs, num_filters_in]
        stride_shape = [1, stride_size[0], stride_size[1], 1]
        input_shape  = tf.shape(inputs)
        output_shape = tf.stack([input_shape[0], input_shape[1], input_shape[2], num_outputs])

        weights = tf.get_variable('weights', kernel_shape, tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        conv_trans = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)
        outputs    = tf.nn.bias_add(conv_trans, bias)
        outputs    = tf.contrib.layers.batch_norm(outputs, center = True, scale = True, is_training = is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def deconv_upsample(inputs, factor, name, padding = 'SAME', activation_fn = None):
    """
    Convolution Transpose upsampling layer with bilinear interpolation weights:
    ISSUE: problems with odd scaling factors
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        factor: Integer, upsampling factor
        name: String, scope name
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height * factor, width * factor, num_filters_in]
    """

    with tf.variable_scope(name):
        stride_shape   = [1, factor, factor, 1]
        input_shape    = tf.shape(inputs)
        num_filters_in = inputs.get_shape()[-1].value
        output_shape   = tf.stack([input_shape[0], input_shape[1] * factor, input_shape[2] * factor, num_filters_in])

        weights = bilinear_upsample_weights(factor, num_filters_in)
        outputs = tf.nn.conv2d_transpose(inputs, weights, output_shape, stride_shape, padding = padding)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def bilinear_upsample_weights(factor, num_outputs):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization:
    ----------
    Args:
        factor: Integer, upsampling factor
        num_outputs: Integer, number of convolution filters

    Returns:
        outputs: Tensor, [kernel_size, kernel_size, num_outputs]
    """

    kernel_size = 2 * factor - factor % 2

    weights_kernel = np.zeros((kernel_size,
                               kernel_size,
                               num_outputs,
                               num_outputs), dtype = np.float32)

    rfactor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = rfactor - 1
    else:
        center = rfactor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]
    upsample_kernel = (1 - abs(og[0] - center) / rfactor) * (1 - abs(og[1] - center) / rfactor)

    for i in xrange(num_outputs):
        weights_kernel[:, :, i, i] = upsample_kernel

    init = tf.constant_initializer(value = weights_kernel, dtype = tf.float32)
    weights = tf.get_variable('weights', weights_kernel.shape, tf.float32, init)

    return weights


def batch_norm(inputs, name, is_training = True, decay = 0.9997, epsilon = 0.001, activation_fn = None):
    """
    Batch normalization layer (currently using Tf-Slim):
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        name: String, scope name
        is_training: Boolean, in training mode or not
        decay: Float, decay rate
        epsilon, Float, epsilon value
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height, width, channels]
    """

    return tf.contrib.layers.batch_norm(inputs, name = name, decay = decay,
                            center = True, scale = True,
                            is_training = is_training,
                            epsilon = epsilon, activation_fn = activation_fn)


def flatten(inputs, name):
    """
    Flatten input tensor:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        name: String, scope name

    Returns:
        outputs: Tensor, [batch_size, height * width * channels]
    """

    with tf.variable_scope(name):
        dim     = inputs.get_shape()[1:4].num_elements()
        outputs = tf.reshape(inputs, [-1, dim])

        return outputs


def fully_connected(inputs, num_outputs, name, activation_fn = tf.nn.relu):
    """
    Fully connected layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        num_outputs: Integer, number of output neurons
        name: String, scope name
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, num_outputs]
    """

    with tf.variable_scope(name):
        num_filters_in = inputs.get_shape()[-1].value

        weights = tf.get_variable('weights', [num_filters_in, num_outputs], tf.float32, xavier_initializer())
        bias    = tf.get_variable('bias', [num_outputs], tf.float32, tf.constant_initializer(0.0))
        outputs = tf.matmul(inputs, weights)
        outputs = tf.nn.bias_add(outputs, bias)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def maxpool(inputs, kernel_size, name, padding = 'SAME'):
    """
    Max pooling layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        kernel_size: List, filter size [height, width]
        name: String, scope name
        stride_size: List, convolution stide [height, width]
        padding: String, input padding

    Returns:
        outputs: Tensor, [batch_size, height / kernelsize[0], width/kernelsize[1], channels]
    """

    kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
    
    outputs = tf.nn.max_pool(inputs, ksize = kernel_shape,
            strides = kernel_shape, padding = padding, name = name)

    return outputs


def dropout(inputs, keep_prob, name):
    """
    Dropout layer:
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        keep_prob: Float, probability of keeping this layer
        name: String, scope name

    Returns:
        outputs: Tensor, [batch_size, height, width, channels]
    """

    return tf.nn.dropout(inputs, keep_prob = keep_prob, name = name)


def concat(inputs1, inputs2, name):
    """
    Concatente two tensors:
    ----------
    Args:
        inputs1: Tensor, [batch_size, height, width, channels]
        inputs2: Tensor, [batch_size, height, width, channels]
        name: String, scope name

    Returns:
        outputs: Tensor, [batch_size, height, width, channels1 + channels2]
    """

    return tf.concat(axis=3, values=[inputs1, inputs2], name = name)


def add(inputs1, inputs2, name, activation_fn = None):
    """
    Add two tensors:
    ----------
    Args:
        inputs1: Tensor, [batch_size, height, width, channels]
        inputs2: Tensor, [batch_size, height, width, channels]
        name: String, scope name
        activation_fn: Tensor fn, activation function on output (can be None)

    Returns:
        outputs: Tensor, [batch_size, height, width, channels]
    """

    with tf.variable_scope(name):
        outputs = tf.add(inputs1, inputs2)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs