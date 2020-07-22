
import numpy as np

#import cv2
#import h5py
#import scipy.io as sio
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pdb
file_id="importantclasses"
#====================================
def labels_predictions_filter_transform(label_test,predictions,class_n,
		debug=1,small_classes_ignore=True,
		important_classes=None,dataset='cv'):
	predictions=predictions.argmax(axis=np.ndim(predictions)-1)
	predictions=np.reshape(predictions,-1)
	label_test=label_test.argmax(axis=np.ndim(label_test)-1)
	label_test=np.reshape(label_test,-1)
	predictions=predictions[label_test<class_n]
	label_test=label_test[label_test<class_n]

	print("========================= Flattened the predictions and labels")	
	print("Loaded predictions unique: ",np.unique(predictions,return_counts=True))
	print("Loaded label test unique: ",np.unique(label_test,return_counts=True))
	
	print("Loaded predictions shape: ",predictions.shape)
	print("Loaded label test shape: ",label_test.shape)

	#pdb.set_trace()
	if small_classes_ignore==True:
		# Eliminate non important classes
		class_list,class_count = np.unique(label_test,return_counts=True)
		if debug>=0: print("Class unique before eliminating non important classes:",class_list,class_count)
		
		if dataset=='cv':
			important_classes_idx=[0,1,2,6,7,8]
		elif dataset=='lm':
			important_classes_idx=[0,1,2,6,8,10,12]

		mode=3
		if mode==1:
			for idx in range(class_n):
				if idx in important_classes_idx and idx in class_list:
					index=int(np.where(class_list==idx)[0])
					if class_count[index]<15000:
						predictions[predictions==idx]=20
						label_test[label_test==idx]=20
				else:
					predictions[predictions==idx]=20
					label_test[label_test==idx]=20
		elif mode==2:
			class_count_min=100000
			important_classes_class_count_min=15000
			#important_classes_class_count_min=1

			#print("Class count min:",class_count_min)

			for idx in range(class_n):
				if idx in class_list:
					class_count_min_idx = important_classes_class_count_min if idx in important_classes_idx else class_count_min
					index=int(np.where(class_list==idx)[0])
					#print("b",index)
					if class_count[index]<class_count_min_idx:
						predictions[predictions==idx]=20
						label_test[label_test==idx]=20
				else:
					predictions[predictions==idx]=20
					label_test[label_test==idx]=20
		elif mode==3: # Just take the important classes, no per-date analysis
			for idx in range(class_n):
				if idx in class_list and idx not in important_classes_idx:
					predictions[predictions==idx]=20
					label_test[label_test==idx]=20




		if debug>=0: print("Class unique after eliminating non important classes:",np.unique(label_test,return_counts=True))
		#print("Pred unique after eliminating non important classes:",np.unique(predictions,return_counts=True))


	if debug>0:
		print("Predictions",predictions.shape)
		print("Label_test",label_test.shape)
	return label_test,predictions
def metrics_get(label_test,predictions,only_basics=False,debug=1, detailed_t=None):
	if debug>0:
		print(predictions.shape,predictions.dtype)
		print(label_test.shape,label_test.dtype)

	metrics={}
	metrics['f1_score']=f1_score(label_test,predictions,average='macro')
	metrics['overall_acc']=accuracy_score(label_test,predictions)
	confusion_matrix_=confusion_matrix(label_test,predictions)
	#print(confusion_matrix_)
	metrics['per_class_acc']=(confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]).diagonal()
	acc=confusion_matrix_.diagonal()/np.sum(confusion_matrix_,axis=1)
	acc=acc[~np.isnan(acc)]
	metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])
	if debug>0:
		print("acc",metrics['per_class_acc'])
		print("Acc",acc)
		print("AA",np.average(acc))
		print("OA",np.sum(confusion_matrix_.diagonal())/np.sum(confusion_matrix_))
		print("AA",metrics['average_acc'])
		print("OA",metrics['overall_acc'])

	if only_basics==False:

		metrics['f1_score_weighted']=f1_score(label_test,predictions,average='weighted')
		        

		metrics['recall']=recall_score(label_test,predictions,average=None)
		metrics['precision']=precision_score(label_test,predictions,average=None)
		if debug>0:
			print(confusion_matrix_.sum(axis=1)[:, np.newaxis].diagonal())
			print(confusion_matrix_.diagonal())
			print(np.sum(confusion_matrix_,axis=1))

			print(metrics)
			print(confusion_matrix_)

			print(metrics['precision'])
			print(metrics['recall'])
	#if detailed_t==6:
	print(confusion_matrix_)

	return metrics


# ======== ConvRNN

path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/cv/densenet/'
prediction_path=path+'prediction.npy'

#prediction_path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/cv/prediction_ConvLSTM_DenseNet_eyesight.npy'

# =========seq2seq 
def experiment_analyze(small_classes_ignore,dataset='cv',
		prediction_filename='prediction_DenseNetTimeDistributed_blockgoer.npy',
		mode='each_date',debug=1):
	path='/home/lvc/Jorg/igarss/convrnn_remote_sensing/results/seq2seq_ignorelabel/'+dataset+'/'

	prediction_path=path+prediction_filename
	predictions=np.load(prediction_path, allow_pickle=True)
	label_test=np.load(path+'labels.npy', allow_pickle=True)

	
	print("Loaded predictions unique: ",np.unique(predictions.argmax(axis=-1),return_counts=True))
	print("Loaded label test unique: ",np.unique(label_test.argmax(axis=-1),return_counts=True))
	
	print("Loaded predictions shape: ",predictions.shape)
	print("Loaded label test shape: ",label_test.shape)

	prediction_unique,prediction_count = np.unique(predictions.argmax(axis=-1),return_counts=True)
	label_test_unique,label_test_count = np.unique(label_test.argmax(axis=-1),return_counts=True)
	print(np.sum(prediction_count[:]))
	print(np.sum(label_test_count[:-1]))
	
	#pdb.set_trace()
	class_n=predictions.shape[-1]

	if mode=='each_date':
		metrics_t={'f1_score':[],'overall_acc':[],
			'average_acc':[]}

		# if dataset=='cv':
		# 	important_classes=[]
		# 	for date in range(14):
		# 		if date<=7:
		# 			date_important_classes=[0,6,8]
		for t in range(label_test.shape[1]):
			predictions_t = predictions[:,t,:,:,:]
			label_test_t = label_test[:,t,:,:,:]

			label_test_t,predictions_t = labels_predictions_filter_transform(
				label_test_t, predictions_t, class_n=class_n,
				debug=debug,small_classes_ignore=small_classes_ignore,
				important_classes=None)
			metrics = metrics_get(label_test_t, predictions_t,
				only_basics=True, debug=debug, detailed_t = t)	
			metrics_t['f1_score'].append(metrics['f1_score'])
			metrics_t['overall_acc'].append(metrics['overall_acc'])
			metrics_t['average_acc'].append(metrics['average_acc'])

		print(metrics_t)
		#pdb.set_trace()
		return metrics_t
	elif mode=='global':
		
		label_test,predictions=labels_predictions_filter_transform(
			label_test,predictions, class_n=class_n)

		print(np.unique(predictions,return_counts=True))
		print(np.unique(label_test,return_counts=True))

		metrics=metrics_get(label_test,predictions)

		return metrics

def experiments_analyze(dataset,experiment_list,mode='each_date'):
	experiment_metrics=[]
	for experiment in experiment_list:
		print("Starting experiment:",experiment)
		experiment_metrics.append(experiment_analyze(
			dataset=dataset,
			prediction_filename=experiment,
			mode=mode,debug=0))
	return experiment_metrics

def experiment_groups_analyze(dataset,experiment_group,
	small_classes_ignore,mode='each_date',exp_id=1):
	save=True
	if save==True:	

		experiment_metrics=[]
		for group in experiment_group:
			group_metrics=[]
			for experiment in group:
				print("Starting experiment:",experiment)
				group_metrics.append(experiment_analyze(
					dataset=dataset,
					prediction_filename=experiment,
					mode=mode,debug=0,
					small_classes_ignore=small_classes_ignore))
			experiment_metrics.append(group_metrics)

	#	for model_id in range(len(experiment_metrics[0])):
	#		for date_id in range(len(experiment_metrics[0][model_id]):

		np.save('experiment_metrics'+str(exp_id)+'.npy',
			experiment_metrics)
	else:
		experiment_metrics=np.load(
			'experiment_metrics'+str(exp_id)+'.npy')
	metrics={}
	total_metrics=[]

	for exp_id in range(len(experiment_metrics[0])):
		exp_id=int(exp_id)
		print(len(experiment_metrics))
		print(len(experiment_metrics[0]))
		print(experiment_metrics[0][0])
		print(experiment_metrics[0][0]['f1_score'])
		#print(experiment_metrics[1][0]['f1_score'])

		print("exp_id",exp_id)		
		metrics['f1_score']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['f1_score'] for i in range(len(experiment_metrics))]),
			axis=0)
		print("metrics f1 score",metrics['f1_score'])
		metrics['overall_acc']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['overall_acc'] for i in range(len(experiment_metrics))]),
			axis=0)

		metrics['average_acc']=np.average(np.asarray(
			[experiment_metrics[int(i)][exp_id]['average_acc'] for i in range(len(experiment_metrics))]),
			axis=0)
		total_metrics.append(metrics.copy())
		print("total metrics f1 score",total_metrics)

	print("metrics['f1_score'].shape",metrics['f1_score'].shape)
	print("total merics len",len(total_metrics))
	print(total_metrics)
	return total_metrics

def experiments_plot(metrics,experiment_list,dataset,
	experiment_id,small_classes_ignore=False):



	if dataset=='cv':
		valid_dates=[0,2,4,5,6,8,10,11,13]
		t_len=len(valid_dates)
	else:
		t_len=len(metrics[0]['f1_score'])

	print("t_len",t_len)
	indices = range(t_len) # t_len
	X = np.arange(t_len)

	exp_id=0
	#width=0.5
	width=0.25
	
	colors=['b','y','c','g','m','r','b','y']
	colors=['b','g','r','c','m','y','b','g']
	#colors=['#7A9AAF','#293C4B','#FF8700']
	#colors=['#4225AC','#1DBBB9','#FBFA17']
	##colors=['b','#FBFA17','c']
	#colors=['#966A51','#202B3F','#DA5534']
	exp_handler=[] # here I save the plot for legend later
	exp_handler2=[] # here I save the plot for legend later
	exp_handler3=[] # here I save the plot for legend later

	figsize=(8,4)
	fig, ax = plt.subplots(figsize=figsize)
	fig2, ax2 = plt.subplots(figsize=figsize)
	fig3, ax3 = plt.subplots(figsize=figsize)

	fig.subplots_adjust(bottom=0.2)
	fig2.subplots_adjust(bottom=0.2)
	fig3.subplots_adjust(bottom=0.2)
	#metrics=metrics[]
	print("Plotting")
	for experiment in experiment_list:

		#print("experiment",experiment)
		print(exp_id)
		metrics[exp_id]['f1_score']=np.transpose(np.asarray(metrics[exp_id]['f1_score']))*100
		metrics[exp_id]['overall_acc']=np.transpose(np.asarray(metrics[exp_id]['overall_acc']))*100
		metrics[exp_id]['average_acc']=np.transpose(np.asarray(metrics[exp_id]['average_acc']))*100
		
		print("Experiment:{}, dataset:{}. Avg F1:{}. Avg OA:{}. Avg AA:{}".format(
			experiment,dataset,
			np.average(metrics[exp_id]['f1_score']),
			np.average(metrics[exp_id]['overall_acc']),
			np.average(metrics[exp_id]['average_acc'])))

		if dataset=='cv':
			
			#print("metrics[exp_id]['average_acc'].shape",
			#	metrics[exp_id]['average_acc'].shape)
			metrics[exp_id]['f1_score']=metrics[exp_id]['f1_score'][valid_dates]
			metrics[exp_id]['overall_acc']=metrics[exp_id]['overall_acc'][valid_dates]
			metrics[exp_id]['average_acc']=metrics[exp_id]['average_acc'][valid_dates]
			#print("metrics[exp_id]['average_acc'].shape",
			#	metrics[exp_id]['average_acc'].shape)
		exp_handler.append(ax.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['f1_score'], 
			color = colors[exp_id], width = width/2))
		ax.set_title('Average F1 Score (%)')
		ax.set_xlabel('Month')
		if dataset=='lm':
			xlim=[-0.5,13] 
			ylim=[10,85]
			xticklabels=['Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
			ax.set_xlim(xlim[0],xlim[1])
			ax3.set_xlim(xlim[0],xlim[1])
			ax.set_ylim(75,100)
			
			if small_classes_ignore==True:
				ax.set_ylim(10,80)
				ax3.set_ylim(70,100)
			else:
				ax.set_ylim(10,82)
				ax3.set_ylim(35,100)
			ax.set_xticks(X+width/2)
			ax.set_xticklabels(xticklabels)
			ax2.set_xticks(X+width/2)
			ax2.set_xticklabels(xticklabels)
			ax3.set_xticks(X+width/2)
			ax3.set_xticklabels(xticklabels)
			
		elif dataset=='cv': 
			xlim=[-0.3,8.9]
			xticklabels=['Oct','Nov','Dec','Jan','Feb','Mar','May','Jun','Jul']

			ax.set_xlim(xlim[0],xlim[1])
			ax3.set_xlim(xlim[0],xlim[1])
			if small_classes_ignore==True:
				ax.set_ylim(40,87)
			else:
				ax.set_ylim(7,75)	
			ax3.set_ylim(30,94)

			ax.set_xticks(X+width/2)
			ax.set_xticklabels(xticklabels)
			ax2.set_xticks(X+width/2)
			ax2.set_xticklabels(xticklabels)
			ax3.set_xticks(X+width/2)
			ax3.set_xticklabels(xticklabels)

		exp_handler2.append(ax2.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['average_acc'], 
			color = colors[exp_id], width = width/2))
		ax2.set_title('Average Accuracy')
		ax2.set_xlabel('Month')
		exp_handler3.append(ax3.bar(X + float(exp_id)*width/2, 
			metrics[exp_id]['overall_acc'], 
			color = colors[exp_id], width = width/2))
		ax3.set_title('Overall Accuracy (%)')
		ax3.set_xlabel('Month')

		#ax3.set_xticks(np.arange(5))
		#ax3.set_xticklabels(('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))
		
		exp_id+=1
	
	#approach1='UConvLSTM'
	#approach2='BConvLSTM'
	#approach3='BDenseConvLSTM'
	
	approach1='BAtrousConvLSTM'
	approach3='BDenseConvLSTM'
	
	approach2='BUnetConvLSTM'
	approach3='BDenseConvLSTM'

	legends=('DeeplabRSConvLSTM','Deeplabv3ConvLSTM','BAtrousConvLSTM','BUnetConvLSTM','BDenseConvLSTM')

	legends=('DeeplabRSDecoderConvLSTM','DeeplabRSConvLSTM','Deeplabv3ConvLSTM','BAtrousConvLSTM','BUnetConvLSTM','BDenseConvLSTM')
	
#	legends=('DeeplabV3+','DeeplabRSDecoder','DeeplabRS','Deeplabv3','BAtrous','BUnet','BDense')
	if experiment_id==1:
		legends=('DeeplabRSDecoder','DeeplabRS','Deeplabv3','BAtrous','BUnet','BDense')
	elif experiment_id==2:
		legends=('BConvLSTM','BDenseConvLSTM','BUnetConvLSTM','BUnet2ConvLSTM','BAtrousGAPConvLSTM','BAtrousConvLSTM','BUnetAtrousConvLSTM','BFCNAtrousConvLSTM')
	elif experiment_id==3:
		legends=('UConvLSTM','BConvLSTM','BUnetConvLSTM','BAtrousConvLSTM')
		legends=('UConvLSTM','BConvLSTM','BDenseConvLSTM','BUnetConvLSTM','BAtrousConvLSTM','BAtrousGAPConvLSTM')
	elif experiment_id==4:
		legends=('UConvLSTM','BConvLSTM','BDenseConvLSTM','BUnetConvLSTM','BAtrousConvLSTM')
	elif experiment_id==6:
		legends=('BConvLSTM','BConvLSTM+WholeInput','UNet_EndConvLSTM','UNet_MidConvLSTM')
	elif experiment_id==6:
		legends=('BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM+Attention')
	elif experiment_id==7:
		legends=('BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM','BUnetConvLSTM+Self attention','Self attention')
	elif experiment_id==8:
		legends=('BConvLSTM','BConvLSTM','BConvLSTM','BConvLSTM','BConvLSTM','BConvLSTM_SelfAttention','BConvLSTM_SelfAttention')
	elif experiment_id==8:
		legends=('BUnetConvlSTM','BUnetStandalone')

	#ncol=len(legends)
	ncol=3

	ax.legend(tuple(exp_handler), legends,loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=ncol)
	ax2.legend(tuple(exp_handler2), legends,loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=ncol)
	ax3.legend(tuple(exp_handler3), legends,loc='lower center', bbox_to_anchor=(0.5, -0.29), shadow=True, ncol=ncol)

	#ax.set_rasterized(True)
	#ax2.set_rasterized(True)
	#ax3.set_rasterized(True)
	
	#fig.savefig("f1_score_"+dataset+".eps",format="eps",dpi=300)
	#fig2.savefig("average_acc_"+dataset+".eps",format="eps",dpi=300)
	#fig3.savefig("overall_acc_"+dataset+".eps",format="eps",dpi=300)
	if small_classes_ignore==True:
		small_classes_ignore_id="_sm"
	else:
		small_classes_ignore_id=""
	fig_names={'f1':"f1_score_importantclasses2_"+dataset+small_classes_ignore_id+".png",
		'aa':"average_acc_importantclasses2_"+dataset+small_classes_ignore_id+".png",
		'oa':"overall_acc_importantclasses2_"+dataset+small_classes_ignore_id+".png"}
	
	for f, filename in zip([fig, fig2, fig3],fig_names.values()):
		f.savefig(filename, dpi=300)

	def fig_crop(inpath):
		fig=cv2.imread(inpath)
		h,w,c =fig.shape
		fig=fig[:,150:w-150,:]
		cv2.imwrite(inpath[:-4]+'_crop.png',fig)

	for k, v in fig_names.items():
		fig_crop(v)
	
	#plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
	#plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
	#plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)


	#plot_metric=[x['f1_score'] for x in metrics]
	#print(plot_metric)
	
	plt.show()

dataset='cv'
load_metrics=False
small_classes_ignore=False
#mode='global'
mode='each_date'
if dataset=='cv':
	experiment_groups=[[
		'prediction_ConvLSTM_seq2seq_batch16_full.npy',
		'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],

		['prediction_ConvLSTM_seq2seq_redoing.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing3.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing3.npy']]
	experiment_groups=[[
		'prediction_pyramid_dilated_bconvlstm_lauras.npy',
		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras.npy',
##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],

		['prediction_pyramid_dilated_bconvlstm_lauras.npy',
		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],

		['prediction_pyramid_dilated_bconvlstm_lauras.npy',
		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
		['prediction_pyramid_dilated_bconvlstm_lauras.npy',
		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]

	experiment_groups=[[
		'prediction_deeplab_rs_nowifi.npy',
		'prediction_deeplabv3_deeplab_param2_2M.npy',
		'prediction_pyramid_dilated_bconvlstm_lauras2.npy',
		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras2.npy',
##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],

		['prediction_deeplab_rs_nowifi.npy',
		'prediction_deeplabv3_deeplab_param2_2M.npy',
		'prediction_pyramid_dilated_bconvlstm_lauras2.npy',
		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],

		['prediction_deeplab_rs_nowifi.npy',
		'prediction_deeplabv3_deeplab_param2_2M.npy',
		'prediction_pyramid_dilated_bconvlstm_lauras2.npy',

		'prediction_deeplab_rs_nowifi.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
		['prediction_deeplab_rs_nowifi.npy',
		'prediction_deeplabv3_deeplab_param2_2M.npy',
		'prediction_pyramid_dilated_bconvlstm_lauras2.npy',

		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]
	exp_id=10 # 4 for thesis and journal

	if exp_id==1:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'prediction_deeplab_rs_multiscale_v3plus.npy',
			'prediction_deeplab_rs_nowifi.npy',
			'prediction_deeplabv3_lauras3.npy',
			'prediction_pyramid_dilated_bconvlstm_lauras2.npy',
			'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras2.npy',
	##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]
	# 		[#'prediction_deeplabv3plus_v3plus2.npy',
	# 		'prediction_deeplab_rs_multiscale_v3plus.npy',
	# 		'prediction_deeplab_rs_nowifi.npy',
	# 		'prediction_deeplabv3_lauras3.npy',
	# 		'prediction_pyramid_dilated_bconvlstm_lauras2.npy',
	# 		'prediction_FCN_ConvLSTM_seq2seq_bi_skip_lauras2.npy',
	# ##		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
	# 		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy']]

	elif exp_id==2:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_BUnetConvLSTM_2convins4.npy',
			'prediction_BAtrousConvLSTM_2convins.npy',
			'prediction_BUnetAtrousConvLSTM_2convins4.npy',
			'prediction_BUnetAtrousConvLSTM_v3p_2convins2.npy']]

		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
			'prediction_BUnetConvLSTM_2convins4.npy',
			'prediction_BUnet2ConvLSTM_raulapproved.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			'prediction_BAtrousConvLSTM_2convins5.npy',
			'prediction_BUnetAtrousConvLSTM_2convins4.npy',
			'prediction_BUnetAtrousConvLSTM_v3p_2convins2.npy'
			]]
	elif exp_id==3:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			#'prediction_BUnetConvLSTM_2convins4.npy',
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_3blocks_repeating3.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BAtrousConvLSTM_2convins5.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			'prediction_BAtrousGAPConvLSTM_repeating3.npy']]

	elif exp_id==4:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
			#'prediction_BUnetConvLSTM_2convins4.npy',
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			#'prediction_BAtrousConvLSTM_2convins5.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			]]

		experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			],

			['prediction_ConvLSTM_seq2seq_redoing.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoing.npy',
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',

			],
			['prediction_ConvLSTM_seq2seq_redoingz.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BAtrousGAPConvLSTM_repeating3.npy',

			],
			['prediction_ConvLSTM_seq2seq_redoingz2.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoing3.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			'prediction_BAtrousGAPConvLSTM_repeating4.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoingz.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoing3.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoing3.npy',
			'prediction_BUnet4ConvLSTM_repeating7.npy',
			'prediction_BAtrousGAPConvLSTM_repeating7.npy',
			]]




	elif exp_id==5:
		experiment_groups=[[
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			'prediction_BAtrousGAPConvLSTM_repeating3.npy',
			'prediction_BAtrousGAPConvLSTM_repeating4.npy',
			'prediction_BAtrousGAPConvLSTM_repeating6.npy',
			]]
		experiment_groups=[[
			'prediction_DenseNetTimeDistributed_128x2_redoing3.npy',
			'prediction_DenseNetTimeDistributed_128x2_3blocks_3blocks_check.npy'
			]]
		experiment_groups=[[
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			#'prediction_BUnet5ConvLSTM_unet_one_conv_in.npy',
			'prediction_BUnet4ConvLSTM_attention_2020.npy',
			
			]]

		#experiment_groups=[[
		#	'prediction_ConvLSTM_seq2seq_bi_redoing3.npy',
		#	'prediction_ConvLSTM_seq2seq_bi_redoing3.npy'
		#	]]
	elif exp_id==6:
		experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_bi_redoing3.npy',
			#'prediction_fcn_bconvlstm.npy',
			'prediction_bconvlstm_wholeinput.npy',
			'prediction_unet_convlstm_temouri2.npy',
			'prediction_BUnet4ConvLSTM_repeating1.npy'
			]]
	elif exp_id==7:
		experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_2020_1.npy',
			'prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_2020_2.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_attention_attention2020.npy'
			
		]]
		experiment_groups=[[
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			'prediction_BUnet4ConvLSTM_SelfAttention_attention_2020.npy',
			#'prediction_BUnet4ConvLSTM_attention_2020.npy',
			'prediction_Unet4ConvLSTM_SelfAttention_attention_2020.npy',
				
		]]
	elif exp_id==8:
		experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoing3.npy',
			'prediction_ConvLSTM_seq2seq_bi_SelfAttention_selfattention.npy',
			'prediction_ConvLSTM_seq2seq_bi_SelfAttention_selfattention2.npy',

				
		]]			
	elif exp_id==9:
		experiment_groups=[[
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BUnet4_Standalone_noconvlstm.npy'
				
		]]	
	elif exp_id==10:
		experiment_groups=[[
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			'prediction_BUnet4ConvLSTM_repeating7.npy'				
		]]	
	elif exp_id==11:
		experiment_groups=[[
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
		]]	

##		'prediction_DenseNetTimeDistributed_128x2_redoing.npy']
		##'prediction_ConvLSTM_seq2seq_loneish.npy',
		##'prediction_ConvLSTM_seq2seq_bi_loneish.npy',
		#'prediction_ConvLSTM_seq2seq_bi_60x2_loneish.npy',
		#'prediction_FCN_ConvLSTM_seq2seq_bi_skip_loneish.npy',
		#'prediction_DenseNetTimeDistributed_blockgoer.npy',
		#'prediction_DenseNetTimeDistributed_128x2_filtersizefix2.npy']
elif dataset=='lm':

	experiment_groups=[[
		'prediction_ConvLSTM_seq2seq_batch16_full.npy',
		'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
		'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'],
		['prediction_ConvLSTM_seq2seq_redoing.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoing.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz.npy'],
		['prediction_ConvLSTM_seq2seq_redoingz2.npy',
		'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
		'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy'],]
	exp_id=4 # choose 4 for thesis and journal paper
	if exp_id==1:
		experiment_groups=[[#'prediction_deeplabv3plus_v3plus2.npy',
			'prediction_deeplab_rs_multiscale_v3plus2.npy',
			'prediction_deeplab_rs_nowifi.npy',
			'prediction_deeplabv3_nowifi.npy',
			'prediction_pyramid_dilated_bconvlstm_v3plus2.npy',
			'prediction_FCN_ConvLSTM_seq2seq_bi_skip_v3plus2.npy', #nowifi
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy']
			]
	elif exp_id==2:
		experiment_groups=[['prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
			'prediction_BUnetConvLSTM_2convins5.npy',
			'prediction_BUnet2ConvLSTM_raulapproved.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			'prediction_BAtrousConvLSTM_2convins5.npy',
			'prediction_BUnetAtrousConvLSTM_2convins5.npy',
			'prediction_BUnetAtrousConvLSTM_v3p_2convins5.npy'		
			]]
	elif exp_id==3:
		experiment_groups=[['prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
			#'prediction_BUnetConvLSTM_2convins5.npy',
			#'prediction_BUnet3ConvLSTM_repeating1.npy',
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BAtrousConvLSTM_2convins5.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			]]
			#'prediction_BUnetAtrousConvLSTM_v3p_2convins2.npy'
	elif exp_id==4:
		experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy',
			'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy',
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoing.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoing.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoing.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BAtrousGAPConvLSTM_repeating6.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoingz.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			'prediction_BAtrousGAPConvLSTM_repeating4.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoingz2.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
			'prediction_BUnet4ConvLSTM_repeating6.npy',
			'prediction_BAtrousGAPConvLSTM_repeating3.npy',
			],
			['prediction_ConvLSTM_seq2seq_redoingz2.npy',
			'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
			'prediction_BUnet4ConvLSTM_repeating7.npy',
			'prediction_BAtrousGAPConvLSTM_repeating7.npy',
			],

			]
	elif exp_id==5:
		experiment_groups=[[
			'prediction_BAtrousGAPConvLSTM_repeating3.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			]]		
		experiment_groups=[[
			#'prediction_BUnet4ConvLSTM_repeating1.npy',
			#'prediction_BUnet4ConvLSTM_repeating2.npy',
			#'prediction_BUnet4ConvLSTM_repeating4.npy',
			#'prediction_BUnet4ConvLSTM_repeating6.npy',
			
			'prediction_BAtrousGAPConvLSTM_repeating3.npy',
			'prediction_BAtrousGAPConvLSTM_raulapproved.npy',
			'prediction_BAtrousGAPConvLSTM_repeating4.npy',
			'prediction_BAtrousGAPConvLSTM_repeating6.npy',
			
			]]		
		experiment_groups=[[
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
			'prediction_DenseNetTimeDistributed_128x2_3blocks_3blocks_check.npy'
			]]		
		experiment_groups=[[
			'prediction_BUnet4ConvLSTM_repeating1.npy',
			'prediction_BUnet4ConvLSTM_repeating2.npy',
			'prediction_BUnet4ConvLSTM_repeating4.npy',
			'prediction_BUnet4ConvLSTM_unet_one_conv_in.npy',
			'prediction_BUnet5ConvLSTM_unet_one_conv_in.npy',
			
			]]


		experiment_groups=[[
			'prediction_DenseNetTimeDistributed_128x2_redoingz2.npy',
			'prediction_DenseNetTimeDistributed_128x2_inconv_unet_one_conv_in.npy'
			#'prediction_DenseNetTimeDistributed_128x2_3blocks_3blocks_check.npy'
			]]
	elif exp_id==6: #hyperparams
		experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_bi_redoingz2.npy',
			'prediction_ConvLSTM_seq2seq_bi_hyperparams.npy' #double filters
			]]
	elif exp_id==7: #Check against matlab f1 results
		experiment_groups=[[
			'prediction_ConvLSTM_seq2seq_batch16_full.npy',
			'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy' #double filters
			]]			
if load_metrics==False:
	experiment_metrics=experiment_groups_analyze(dataset,experiment_groups,
		mode=mode,exp_id=exp_id,small_classes_ignore=small_classes_ignore)
	np.save("experiment_metrics_"+dataset+"_"+file_id+".npy",experiment_metrics)

else:
	experiment_metrics=np.load("experiment_metrics_"+dataset+"_"+file_id+".npy")
	print("Difference F1 in percentage",np.average(experiment_metrics[2]['f1_score']-experiment_metrics[1]['f1_score']))
	print("Difference OA in percentage",np.average(experiment_metrics[2]['overall_acc']-experiment_metrics[1]['overall_acc']))


if mode=='each_date':
	experiments_plot(experiment_metrics,experiment_groups[0],
		dataset,experiment_id=exp_id,
		small_classes_ignore=small_classes_ignore)

#metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])]


