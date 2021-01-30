# This is the code for the paper "Fully Convolutional Recurrent Networks for Multidate Crop Recognition from Multitemporal Image Sequence"

This is a first version which will be optimized soon.

## Preparing the input images 

Use the helper file tif_to_npy.py to convert the TIF VH and VV input image bands for each date into NPY format, while also converting from dB to intensity values. Set the im_folders variable to a list of folders for each image date.

## Instructions

To train Campo Verde dataset and BUnetConvLSTM network:

1. Copy the sequence of input images in "dataset/dataset/cv_data/in_np2/" folder. Rename the input images as 'inX.npy', where X is an integer representing the image ID in the sequence.
2. Copy the sequence of output label images in "dataset/dataset/cv_data/labels/" folder. Rename the label images as 'X.tif', where X is an integer representing the image ID in the sequence.
3. Execute "cd networks/convlstm_networks/train_src/scripts/"
4. Execute ". experiment_automation_lv2.sh"


## Specify the execution order durinig training (select if you want to extract image patches and select the network)

The file "experiment_automation_lv2.sh" specifies the execution order during training. For example, the order could be 1. extract image patches and 2. train BUnetConvLSTM network (Default configuration).

Modify this script to train other networks: 
  1. open the "experiment_automation_lv2.sh" script, 
  2. In this script, change the second parameter from the "experiment_automation.sh" command, which is initially set to BUnet4ConvLSTM, to:
    - ConvLSTM_seq2seq (In the paper known as UConvLSTM)
    - ConvLSTM_seq2seq_bi (In the paper known as BConvLSTM)
    - DenseNetTimeDistributed_128x2 (In the paper known as BDenseConvLSTM)
    - BUnet4ConvLSTM (In the paper known as BUnetConvLSTM)
    - BAtrousGAPConvLSTM (In the paper known as BAtrousGAPConvLSTM)
