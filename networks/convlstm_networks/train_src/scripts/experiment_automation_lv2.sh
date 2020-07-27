KERAS_BACKEND=tensorflow
id='lmish'

#dataset='campo_verde'
dataset='cv'


# ==== EXTRACT PATCHES
. patches_extract.sh $dataset
# ===== USE MODEL



. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset  # Unet5 uses 1 conv. in
#. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  # Unet5 uses 1 conv. in
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
#. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  # gonna test balancing after replication
#. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  # Unet5 uses 1 conv. in

