# ============================================================== #
#                        Dataset Loader                          #
#                                                                #
#                                                                #
# Processing occurs on a single image at a time. Images are      #
# read and preprocessed in parallel across multiple threads. A   #
# Batch is then formed from these data to be used for training   #
# or evaluation                                                  #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf


IMAGE_FORMAT = 'PNG'


def inputs(data_files, train, batch_size, image_size, 
                     num_epochs, num_preprocess_threads = 4):
    """
    Generate shuffled batches from dataset images:
    ----------
    Args:
        data_files: string, array of shared tensor-records
        train: boolean, is in training mode (shuffle data)
        batch_size: integer, number of examples in batch
        image_size: integer, size used to resize loaded image (w & h)
        num_epochs: integer, number of epochs
        num_preprocess_threads: integer, total number of preprocessing threads

    Returns:
        images: Colored images. 4D tensor of size [batch_size, image_size, image_size, 3]
        labels: Label images. 4D tensor of size [batch_size, image_size, image_size, 1]
        filenames: 1-D string Tensor of [batch_size]
    """

    images, labels, filenames = batch_inputs(data_files = data_files, 
                    batch_size = batch_size, image_size = image_size,
                    train = train, num_epochs = num_epochs,
                    num_preprocess_threads = num_preprocess_threads)

    return images, labels, filenames


def image_preprocessing(image, image_size, is_color, scope = None):
    """
    Process & resized one image:
    ----------
    Args:
        image: 3-D float Tensor
        image_size: integer
        is_color: is color image (3 channels)
        scope: optional scope

    Returns:
        image: resized 3-D float Tensor
    """

    with tf.name_scope(scope, 'process_image', [image, image_size, image_size]):
        image = tf.expand_dims(image, 0)

        if is_color:
            image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
        else:
            image = tf.image.resize_nearest_neighbor(image, [image_size, image_size], align_corners=False)

        image = tf.squeeze(image, [0])

    return image


def parse_example_proto(example_serialized, image_format):
    """
    Parses an Example proto containing a training example of an image
    and decode image content according to their corresponding format:
    ----------
    Args:
        example_serialized: scalar Tensor tf.string containing a serialized
        image_format: string, image format used for decoding

    Returns:
        color_image: Tensor decoded color image
        label_image: Tensor decoded label image
        filename: Tensor file name
    """

    feature_map = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/format': tf.FixedLenFeature([], dtype = tf.string, default_value = ''),
        'image/encoded/color': tf.FixedLenFeature([], dtype = tf.string, default_value = ''),
        'image/encoded/label': tf.FixedLenFeature([], dtype = tf.string, default_value = ''),
        'image/filename': tf.FixedLenFeature([], dtype = tf.string, default_value = '')
    }

    features     = tf.parse_single_example(example_serialized, feature_map)
    height       = tf.cast(features['image/height'], dtype = tf.int32)
    width        = tf.cast(features['image/width'], dtype = tf.int32)
    filename     = tf.cast(features['image/filename'], dtype = tf.string)
    
    if image_format.lower().endswith(('png')):
      color_image = tf.image.decode_png(features['image/encoded/color'])
      label_image = tf.image.decode_png(features['image/encoded/label'])
    else:
      color_image = tf.image.decode_jpeg(features['image/encoded/color'])
      label_image = tf.image.decode_jpeg(features['image/encoded/label'])

    color_shape = tf.stack([height, width, 3])
    label_shape = tf.stack([height, width, 1])

    color_image = tf.reshape(color_image, color_shape)
    label_image = tf.reshape(label_image, label_shape)

    return color_image, label_image, filename


def batch_inputs(data_files, batch_size, image_size, train, num_epochs, num_preprocess_threads):
    """
    Contruct batches of training or evaluation examples from the image dataset:
    ----------
    Args:
        data_files: string, array of shared tensor-records
        batch_size: integer, size of each batch
        image_size: integer, size used to resize loaded image (w & h)
        num_epochs: integer, number of epochs
        num_preprocess_threads: integer, total number of preprocessing threads

    Returns:
        images: 4-D float Tensor of a batch of resized color images
        labels: 4-D float Tensor of a batch of resized label images
        filenames: 1-D string Tensor of a batch of image filenames
    """

    with tf.name_scope('batch_processing'):
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle = True, num_epochs = num_epochs)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle = False, num_epochs = num_epochs)

        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

        color_image, label_image, filename = parse_example_proto(example_serialized, IMAGE_FORMAT)
        color_image = image_preprocessing(color_image, image_size, is_color = True)
        label_image = image_preprocessing(label_image, image_size, is_color = False)

        color_image = tf.cast(color_image, tf.float32)
        label_image = tf.cast(label_image, tf.float32)

        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        
        if train:
            images, labels, filenames = tf.train.shuffle_batch(
                [color_image, label_image, filename],
                batch_size = batch_size, num_threads = num_preprocess_threads,
                capacity = capacity,
                min_after_dequeue = min_after_dequeue)
        else:
            # Don't shuffle batches when testing
            images, labels, filenames = tf.train.batch(
                [color_image, label_image, filename],
                batch_size = batch_size, num_threads = num_preprocess_threads,
                capacity = capacity)

        classes = tf.reshape(classes, [batch_size])

        return images, labels, filenames
