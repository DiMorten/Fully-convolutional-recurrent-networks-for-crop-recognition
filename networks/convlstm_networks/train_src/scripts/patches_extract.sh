dataset=$1 # could be cv or lem


if [ "$dataset" == "cv_seq1" ]
then
	dataset_path="../cv_data/"
	t_len=7
	im_h=8492
	im_w=7995
	class_n=12 #11+bcknd
elif [ "$dataset" == "cv" ]
then
	dataset_path="../cv_data/"
	t_len=14
	im_h=8492
	im_w=7995
	class_n=12 #11+bcknd 
else
	dataset_path="../lm_data/"
	t_len=13
	im_w=8658
	im_h=8484
	class_n=15 #14+bcknd
fi

# ==== EXTRACT PATCHES
#cd ~/Jorg/deep_learning/LSTM-Final-Project/src_seq2seq
cd ../../../../dataset/dataset/patches_extract_script

#python patches_store.py -ttmn="TrainTestMask.tif" -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=$class_n --log_dir="../data/summaries/" --path="${dataset_path}" --im_h=$im_h --im_w=$im_w --band_n=2 --t_len=$t_len --id_first=1 -tof=False -nap=10000 -psv=True
python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=$class_n --log_dir="../data/summaries/" --path="${dataset_path}" --im_h=$im_h --im_w=$im_w --band_n=2 --t_len=$t_len --id_first=1 -tof=False -nap=10000 -psv=True

#python patches_store.py -ttmn="TrainTestMaskISPRSReview.png" -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=$class_n --log_dir="../data/summaries/" --path="${dataset_path}" --im_h=$im_h --im_w=$im_w --band_n=2 --t_len=$t_len --id_first=1 -tof=False -nap=10000 -psv=True

#python patches_store.py -mm="ram" --debug=1 -pl 32 -po 0 -ts 0 -tnl 10000 -bs=5000 --batch_size=128 --filters=256 -m="smcnn_semantic" --phase="repeat" -sc=False --class_n=12 --log_dir="../data/summaries/" --path="../cv_data/" --im_h=8492 --im_w=7995 --band_n=2 --t_len=7 --id_first=1 -tof=False -nap=10000 -psv=True

#cd ~/Jorg/igarss/convrnn_remote_sensing/src_seq2seq_ignorelabel/scripts
cd ../../../networks/convlstm_networks/train_src/scripts

#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' 
#. experiment_automation.sh $id 'ConvLSTM_seq2seq' 
#. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi_60x2' 
#. experiment_automation.sh $id 'FCN_ConvLSTM_seq2seq_bi_skip' $dataset




