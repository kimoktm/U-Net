# ============================================================== #
#                          Image Coder                           #
#                                                                #
#                                                                #
# Helper class that provides TensorFlow image coding utilities   #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

import tensorflow as tf


class ImageCoder(object):

    def __init__(self):
        # Create a single Session to run all image coding calls
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data
        self._png_data = tf.placeholder(dtype = tf.string)
        png_image = tf.image.decode_png(self._png_data)
        self._png_to_jpeg = tf.image.encode_jpeg(png_image)

        # Initializes function that converts JPEG to PNG data
        self._jpeg_data = tf.placeholder(dtype = tf.string)
        jpeg_image = tf.image.decode_jpeg(self._jpeg_data)
        self._jpeg_to_png = tf.image.encode_png(jpeg_image)

        # Initializes function that decodes JPEG data
        self._decode_jpeg_data = tf.placeholder(dtype = tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data)

        # Initializes function that decodes PNG data
        self._decode_png_data = tf.placeholder(dtype = tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data)


    def is_png(self, filename):
        """
        Determine if a file contains a PNG format image:
        ----------
        Args:
            filename: string, path of the image file

        Returns:
            boolean indicating if the image is a PNG
        """

        return filename.lower().endswith(('png'))


    def is_jpeg(self, filename):
        """
        Determine if a file contains a JPG format image:
        ----------
        Args:
            filename: string, path of the image file

        Returns:
            boolean indicating if the image is a PNG
        """

        return filename.lower().endswith(('jpg', 'jpeg'))


    def png_to_jpeg(self, image_data):
        """
        Convert PNG data fromat to JPEG format:
        ----------
        Args:
            image_data: PNG data

        Returns:
            JPEG data
        """

        return self._sess.run(self._png_to_jpeg,
                             feed_dict={self._png_data: image_data})


    def jpeg_to_png(self, image_data):
        """
        Convert JPEG data fromat to PNG format:
        ----------
        Args:
            image_data: JPEG data

        Returns:
            PNG data
        """

        return self._sess.run(self._jpeg_to_png,
                             feed_dict={self._jpeg_data: image_data})


    def decode_jpeg(self, image_data):
        """
        Decode raw data to JPEG data:
        ----------
        Args:
            image_data: Raw image data

        Returns:
            JPEG data
        """

        return self._sess.run(self._decode_jpeg,
                             feed_dict={self._decode_jpeg_data: image_data})


    def decode_png(self, image_data):
        """
        Decode raw data to PNG data:
        ----------
        Args:
            image_data: Raw image data

        Returns:
            PNG data
        """

        return self._sess.run(self._decode_png,
                             feed_dict={self._decode_png_data: image_data})
