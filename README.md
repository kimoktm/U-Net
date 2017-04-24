# Unet
Semantic Segmentation neural net based on Unet [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Batch norms and dropouts are added to the network as well as weighted cross entropy loss for multi-class segmentation.

<img src="Images/framework.png" width="800px"/>

### Dependencies
- python 2.7
- [TensorFlow >=1.0.0](https://www.tensorflow.org/get_started/os_setup)
- In addition, please `pip install -r requirements.txt` to install the following packages:
    - `Pillow`
    - `numpy`
    - `tensorflow>=1.0.0`

### Data Preprocessing
Tensor records are used as means of data storage for ease of use and distribution. To convert a normal dataset to tfrecords use `data/dataset_to_tfrecords.py`. The dataset images should be in the same folder (im1_color.png, im1_label.png) with PNG or JPG format. The label images must be 1 channel images.

Convert each dataset (training, testing) to tfrecords `Datasets/tfrecords` using

  ```
  python data/dataset_to_tfrecords.py --data_dir Datasets/training --output_dir Datasets/tfrecords --name_color _image --name_label _label
  ```

name_color, name_label specifies the sffuix of the image presenting the RGB and labels respectively.

### Training
- To train Unet run `unet_train` passing tfrecords dir
   
    ```
    python unet_train.py --tfrecords_dir ../Datasets/tfrecords  --checkpoint_dir ../Datasets/checkpoints
    ```
  
  Use help to check the other parameters
    ```
    python unet_train.py -h
    ```

### Evaluation
- To evaluate Unet run `unet_eval` passing tfrecords dir

    ```
    python unet_eval.py --tfrecords_dir ../Datasets/tfrecords
    ```

    To save the predicted annotations as png files, pass in an output directory to the eval script

    ```
    python unet_eval.py  --tfrecords_dir ../Datasets/tfrecords --output_dir predictions
    ```

### Citing Unet
Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical
image segmentation. In: International Conference on Medical Image Computing
and Computer-Assisted Intervention. pp. 234–241. Springer (2015) [pdf](https://arxiv.org/abs/1505.04597).

    @inproceedings{fusenet2016accv,
     author    = "Olaf Ronneberger, Philipp Fischer, and Thomas Brox",
     title     = "U-Net: Convolutional Networks for Biomedical Image Segmentation",
     booktitle = "Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015",
     year      = "2015",
     month     = "October",
    }
