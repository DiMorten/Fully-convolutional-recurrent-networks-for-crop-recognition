# This is the code for the paper "Fully Convolutional Recurrent Networks for Multidate Crop Recognition from Multitemporal Image Sequence"

This is an unpolished version which will be optimized soon.

## Instructions

To train Campo Verde dataset and BUnetConvLSTM network:

1. Copy the sequence of input images in "dataset/dataset/cv_data/in_np2/" folder. Rename the input images as 'inX.tif', where X is an integer representing the image ID in the sequence.
2. Copy the sequence of output label images in "dataset/dataset/cv_data/labels/" folder. Rename the label images as 'X.tif', where X is an integer representing the image ID in the sequence.
3. Execute "cd networks/convlstm_networks/train_src/scripts/"
4. Execute ". experiment_automation_lv2.sh"

## Select the network to train

The file "experiment_automation_lv2.sh" specifies the networks to be trained. For example, the order could be 1. extract image patches and 2. train BUnetConvLSTM network (Default configuration).

Modify this script to train other networks: 
  1. open the "experiment_automation_lv2.sh" script, 
  2. change the "id" parameter from the "experiment_automation.sh" command to:
    - ConvLSTM_seq2seq
    - ConvLSTM_seq2seq_bi
    - DenseNetTimeDistributed_128x2
    - BUnet4ConvLSTM
    - BAtrousGAPConvLSTM
