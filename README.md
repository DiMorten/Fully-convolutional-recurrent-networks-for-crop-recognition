This is the code for the paper "Fully Convolutional Recurrent Networks for Crop Recognition Using Image Sequences"

This is an unpolished version which will be optimized soon.

Instructions:

To train Campo Verde dataset and BUnetConvLSTM network:

1. Copy the sequence of input images in "deep_learning/LSTM-Final-Project/cv_data/in_np2/" folder
2. Copy the sequence of output label images in "deep_learning/LSTM-Final-Project/cv_data/labels/" folder
3. Execute "cd igarss/convrnn_remote_sensing/src_seq2seq_ignorelabel/scripts/"
4. Execute ". experiment_automation_lv2.sh"

=========================================

The file "experiment_automation_lv2.sh" specifies the order of execution. For example, the order could be 1. extract image patches and 2. train BUnetConvLSTM network (Default configuration).

Modify this script to train other networks: 
  1. open the "experiment_automation_lv2.sh" script, 
  2. change the "id" parameter from the "experiment_automation.sh" command to:
    - ConvLSTM_seq2seq
    - ConvLSTM_seq2seq_bi
    - DenseNetTimeDistributed_128x2
    - BUnet4ConvLSTM
    - BAtrousGAPConvLSTM
