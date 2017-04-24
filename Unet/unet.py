# ============================================================== #
#                            U-net                               #
#                                                                #
#                                                                #
# Unet tensorflow implementation                                 #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

import tensorflow as tf

import utils.layers as layers


def build(color_inputs, num_classes, is_training):
    """
    Build unet network:
    ----------
    Args:
        color_inputs: Tensor, [batch_size, height, width, 3]
        num_classes: Integer, number of segmentation (annotation) labels
        is_training: Boolean, in training mode or not (for dropout & bn)
    Returns:
        logits: Tensor, predicted annotated image flattened 
                              [batch_size * height * width,  num_classes]
    """

    dropout_keep_prob = tf.where(is_training, 0.2, 1.0)

    # Encoder Section
    # Block 1
    color_conv1_1 = layers.conv_btn(color_inputs,  [3, 3], 64, 'conv1_1', is_training = is_training)
    color_conv1_2 = layers.conv_btn(color_conv1_1, [3, 3], 64, 'conv1_2', is_training = is_training)
    color_pool1   = layers.maxpool(color_conv1_2, [2, 2],  'pool1')

    # Block 2
    color_conv2_1 = layers.conv_btn(color_pool1,   [3, 3], 128, 'conv2_1', is_training = is_training)
    color_conv2_2 = layers.conv_btn(color_conv2_1, [3, 3], 128, 'conv2_2', is_training = is_training)
    color_pool2   = layers.maxpool(color_conv2_2, [2, 2],   'pool2')

    # Block 3
    color_conv3_1 = layers.conv_btn(color_pool2,   [3, 3], 256, 'conv3_1', is_training = is_training)
    color_conv3_2 = layers.conv_btn(color_conv3_1, [3, 3], 256, 'conv3_2', is_training = is_training)
    color_pool3   = layers.maxpool(color_conv3_2, [2, 2],   'pool3')
    color_drop3   = layers.dropout(color_pool3, dropout_keep_prob, 'drop3')

    # Block 4
    color_conv4_1 = layers.conv_btn(color_drop3,   [3, 3], 512, 'conv4_1', is_training = is_training)
    color_conv4_2 = layers.conv_btn(color_conv4_1, [3, 3], 512, 'conv4_2', is_training = is_training)
    color_pool4   = layers.maxpool(color_conv4_2, [2, 2],   'pool4')
    color_drop4   = layers.dropout(color_pool4, dropout_keep_prob, 'drop4')

    # Block 5
    color_conv5_1 = layers.conv_btn(color_drop4,   [3, 3], 1024, 'conv5_1', is_training = is_training)
    color_conv5_2 = layers.conv_btn(color_conv5_1, [3, 3], 1024, 'conv5_2', is_training = is_training)
    color_drop5   = layers.dropout(color_conv5_2, dropout_keep_prob, 'drop5')

    # Decoder Section
    # Block 1
    upsample6     = layers.deconv_upsample(color_drop5, 2,  'upsample6')
    concat6       = layers.concat(upsample6, color_conv4_2, 'contcat6')
    color_conv6_1 = layers.conv_btn(concat6,       [3, 3], 512, 'conv6_1', is_training = is_training)
    color_conv6_2 = layers.conv_btn(color_conv6_1, [3, 3], 512, 'conv6_1', is_training = is_training)
    color_drop6   = layers.dropout(color_conv6_2, dropout_keep_prob, 'drop6')

    # Block 2
    upsample7     = layers.deconv_upsample(color_drop6, 2,  'upsample7')
    concat7       = layers.concat(upsample7, color_conv3_2, 'concat7')
    color_conv7_1 = layers.conv_btn(concat7,       [3, 3], 256, 'conv7_1', is_training = is_training)
    color_conv7_2 = layers.conv_btn(color_conv7_1, [3, 3], 256, 'conv7_1', is_training = is_training)
    color_drop7   = layers.dropout(color_conv7_2, dropout_keep_prob, 'drop7')

    # Block 3
    upsample8     = layers.deconv_upsample(color_drop7, 2,  'upsample8')
    concat8       = layers.concat(upsample8, color_conv2_2, 'concat8')
    color_conv8_1 = layers.conv_btn(concat8,       [3, 3], 128, 'conv8_1', is_training = is_training)
    color_conv8_2 = layers.conv_btn(color_conv8_1, [3, 3], 128, 'conv8_1', is_training = is_training)

    # Block 4
    upsample9     = layers.deconv_upsample(color_conv9_2, 2, 'upsample9')
    concat9       = layers.concat(upsample8, color_conv1_2,  'concat9')
    color_conv9_1 = layers.conv_btn(concat9,       [3, 3], 64,   'conv9_1', is_training = is_training)
    color_conv9_2 = layers.conv_btn(color_conv9_1, [3, 3], 64,   'conv9_1', is_training = is_training)

    # Block 5
    score  = layers.conv(color_conv9_2, [1, 1], num_classes, 'score', activation_fn = None)
    logits = tf.reshape(score, (-1, num_classes))

    return logits


def segmentation_loss(logits, labels, class_weights = None):
    """
    Segmentation loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size * height * width, num_classes]
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None

    Returns:
        segment_loss: Segmentation loss
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels = labels, logits = logits, name = 'segment_cross_entropy_per_example')

    if class_weights is not None:
        weights = tf.matmul(labels, class_weights, a_is_sparse = True)
        weights = tf.reshape(weights, [-1])
        cross_entropy = tf.multiply(cross_entropy, weights)

    segment_loss  = tf.reduce_mean(cross_entropy, name = 'segment_cross_entropy')

    tf.summary.scalar("loss/segmentation", segment_loss)

    return segment_loss


def l2_loss():
    """
    L2 loss:
    -------
    Returns:
        l2_loss: L2 loss for all weights
    """
    
    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

    tf.summary.scalar("loss/weights", l2_loss)

    return l2_loss


def loss(logits, labels, weight_decay_factor, class_weights = None):
    """
    Total loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size, height, width, 1]
        weight_decay_factor: float, factor with which weights are decayed
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None

    Returns:
        total_loss: Segmentation + Classification losses + WeightDecayFactor * L2 loss
    """

    segment_loss = segmentation_loss(logits, labels, class_weights)
    total_loss   = segment_loss + weight_decay_factor * l2_loss()

    tf.summary.scalar("loss/total", total_loss)

    return total_loss


def accuracy(logits, labels):
    """
    Segmentation accuracy:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size, height, width, 1]

    Returns:
        segmentation_accuracy: Segmentation accuracy
    """

    labels = tf.to_int64(labels)
    labels = tf.reshape(labels, [-1, 1])
    predicted_annots = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])
    correct_predictions = tf.equal(predicted_annots, labels)
    segmentation_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    tf.summary.scalar("accuarcy/segmentation", segmentation_accuracy)

    return segmentation_accuracy


def train(loss, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, global_step):
    """
    Train opetation:
    ----------
    Args:
        loss: loss to use for training
        learning_rate: Float, learning rate
        learning_rate_decay_steps: Int, amount of steps after which to reduce the learning rate
        learning_rate_decay_rate: Float, decay rate for learning rate

    Returns:
        train_op: Training operation
    """
    
    decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 
                            learning_rate_decay_steps, learning_rate_decay_rate, staircase = True)

    # execute update_ops to update batch_norm weights
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer   = tf.train.AdamOptimizer(decayed_learning_rate)
        train_op    = optimizer.minimize(loss, global_step = global_step)

    tf.summary.scalar("learning_rate", decayed_learning_rate)

    return train_op


def predict(logits, batch_size, image_size):
    """
    Prediction operation:
    ----------------
    Args:
        logits: Tensor, predicted    [batch_size * height * width, num_classes]
        batch_size: Int, batch size
        image_size: Int, image width/height
    
    Returns:
        predicted_images: Tensor, predicted images   [batch_size, image_size, image_size]
    """

    predicted_images = tf.reshape(tf.argmax(logits, axis = 1), [batch_size, image_size, image_size])

    return predicted_images