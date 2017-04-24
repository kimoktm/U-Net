# ============================================================== #
#                         Fusnet eval                            #
#                                                                #
#                                                                #
# Eval fusnet with processed dataset in tfrecords format         #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf
from PIL import Image

import os
import glob
import argparse

import data.dataset_loader as dataset_loader
import unet
import numpy as np

# Basic model parameters as external flags.
FLAGS = None


def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '%s-*' % type)
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size


def maybe_save_images(images, filenames):
    """
    Save images to disk
    -------------
    Args:
        images: numpy array     [batch_size, image_size, image_size]
        filenames: numpy string array, filenames corresponding to the images   [batch_size]
    """

    if FLAGS.output_dir is not None:
        batch_size = images.shape[0]
        for i in xrange(batch_size):
            image_array = images[i, :, :]
            file_path = os.path.join(FLAGS.output_dir, filenames[i])
            image = Image.fromarray(np.uint8(image_array))
            image.save(file_path)
    
    
def evaluate():
    """
    Eval unet using specified args:
    """

    data_files, data_size = load_datafiles(FLAGS.tfrecords_prefix)
    images, labels, filenames = dataset_loader.inputs(
                                    data_files = data_files,
                                    image_size = FLAGS.image_size,
                                    batch_size = FLAGS.batch_size,
                                    num_epochs = 1,
                                    train = False)

    logits = unet.build(images, FLAGS.num_classes, False)

    predicted_images = unet.predict(logits, FLAGS.batch_size, FLAGS.image_size)

    accuracy = unet.accuracy(logits, labels)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    saver = tf.train.Saver()

    if not tf.gfile.Exists(FLAGS.checkpoint_path + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        print('[INFO    ]\tFound checkpoint file, restoring model.')
        saver.restore(sess, FLAGS.checkpoint_path)
    
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    global_accuracy = 0.0

    step = 0
    
    try:
    
        while not coord.should_stop():
            acc_seg_value, predicted_images_value, filenames_value = sess.run([accuracy, predicted_images, filenames])
            global_accuracy += acc_seg_value

            maybe_save_images(predicted_images_value, filenames_value)
            print('[PROGRESS]\tAccuracy for current batch: %.5f' % (acc_seg_value))
            step += 1

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone evaluating in %d steps.' % step)

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    global_accuracy = global_accuracy / step

    print('[RESULT  ]\tGlobal accuracy = %.5f' % (global_accuracy))

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(_):
    """
    Run unet prediction on input tfrecords
    """

    if FLAGS.output_dir is not None:
        if not tf.gfile.Exists(FLAGS.output_dir):
            print('[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(FLAGS.output_dir)
        
    evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Eval Unet on given tfrecords.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'training')
    parser.add_argument('--checkpoint_path', help = 'Path of checkpoint to restore. (Ex: ../Datasets/checkpoints/unet.ckpt-80000)')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 2)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 2)
    parser.add_argument('--output_dir', help = 'Output directory for the prediction files. If this is not set then predictions will not be saved')
    
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
