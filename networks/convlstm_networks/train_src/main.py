from utils import *
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam,Adagrad 
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import sys
import glob

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
# Local
from densnet import DenseNetFCN
from densnet_timedistributed import DenseNetFCNTimeDistributed

from metrics import fmeasure,categorical_accuracy
import deb
from keras_weighted_categorical_crossentropy import weighted_categorical_crossentropy, sparse_accuracy_ignoring_last_label, weighted_categorical_crossentropy_ignoring_last_label
from keras.models import load_model
from keras.layers import ConvLSTM2D, ConvGRU2D, UpSampling2D, multiply
from keras.utils.vis_utils import plot_model
from keras.regularizers import l1,l2
import time
import pickle
from keras_self_attention import SeqSelfAttention
import pdb
parser = argparse.ArgumentParser(description='')
parser.add_argument('-tl', '--t_len', dest='t_len',
					type=int, default=7, help='t len')
parser.add_argument('-cn', '--class_n', dest='class_n',
					type=int, default=11, help='class_n')
parser.add_argument('-chn', '--channel_n', dest='channel_n',
					type=int, default=2, help='channel number')

parser.add_argument('-pl', '--patch_len', dest='patch_len',
					type=int, default=32, help='patch len')
parser.add_argument('-pstr', '--patch_step_train', dest='patch_step_train',
					type=int, default=32, help='patch len')
parser.add_argument('-psts', '--patch_step_test', dest='patch_step_test',
					type=int, default=None, help='patch len')

parser.add_argument('-db', '--debug', dest='debug',
					type=int, default=1, help='patch len')
parser.add_argument('-ep', '--epochs', dest='epochs',
					type=int, default=8000, help='patch len')
parser.add_argument('-pt', '--patience', dest='patience',
					type=int, default=10, help='patience')

parser.add_argument('-bstr', '--batch_size_train', dest='batch_size_train',
					type=int, default=32, help='patch len')
parser.add_argument('-bsts', '--batch_size_test', dest='batch_size_test',
					type=int, default=32, help='patch len')

parser.add_argument('-em', '--eval_mode', dest='eval_mode',
					default='metrics', help='Test evaluate mode: metrics or predict')
parser.add_argument('-is', '--im_store', dest='im_store',
					default=True, help='Store sample test predicted images')
parser.add_argument('-eid', '--exp_id', dest='exp_id',
					default='default', help='Experiment id')

parser.add_argument('-path', '--path', dest='path',
					default='../data/', help='Experiment id')
parser.add_argument('-mdl', '--model_type', dest='model_type',
					default='DenseNet', help='Experiment id')


args = parser.parse_args()

if args.patch_step_test==None:
	args.patch_step_test=args.patch_len

deb.prints(args.patch_step_test)

def model_summary_print(s):
	with open('model_summary.txt','w+') as f:
		print(s, file=f)


def txt_append(filename, append_text):
	with open(filename, "a") as myfile:
		myfile.write(append_text)
def load_obj(name ):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)
# ================= Generic class for init values =============================================== #
class NetObject(object):

	def __init__(self, patch_len=32, patch_step_train=32,patch_step_test=32, path="../data/", im_name_train="Image_Train.tif", im_name_test="Image_Test.tif", label_name_train="Reference_Train.tif", label_name_test="Reference_Test.tif", channel_n=2, debug=1,exp_id="skip_connections",
		t_len=7,class_n=11):
		self.patch_len = patch_len
		self.path = {"v": path, 'train': {}, 'test': {}}
		self.image = {'train': {}, 'test': {}}
		self.patches = {'train': {}, 'test': {}}

		self.patches['train']['step']=patch_step_train
		self.patches['test']['step']=patch_step_test        
		self.path['train']['in'] = path + 'train_test/train/ims/'
		self.path['test']['in'] = path + 'train_test/test/ims/'
		self.path['train']['label'] = path + 'train_test/train/labels/'
		self.path['test']['label'] = path + 'train_test/test/labels/'
		self.channel_n = channel_n
		self.debug = debug
		self.class_n = class_n
		self.report={'best':{}, 'val':{}}
		self.report['exp_id']=exp_id
		self.report['best']['text_name']='result_'+exp_id+'.txt'
		self.report['best']['text_path']='../results/'+self.report['best']['text_name']
		self.report['best']['text_history_path']='../results/'+'history.txt'
		self.report['val']['history_path']='../results/'+'history_val.txt'
		
		self.t_len=t_len

# ================= Dataset class implements data loading, patch extraction, metric calculation and image reconstruction =======#
class Dataset(NetObject):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.im_gray_idx_to_rgb_table=[[0,[0,0,255],29],
									[1,[0,255,0],150],
									[2,[0,255,255],179],
									[3,[255,255,0],226],
									[4,[255,255,255],255]]
		if self.debug >= 1:
			print("Initializing Dataset instance")

	def create(self):
		self.image["train"], self.patches["train"] = self.subset_create(
			self.path['train'],self.patches["train"]['step'])
		self.image["test"], self.patches["test"] = self.subset_create(
			self.path['test'],self.patches["test"]['step'])

		if self.debug:
			deb.prints(self.image["train"]['in'].shape)
			deb.prints(self.image["train"]['label'].shape)

			deb.prints(self.image["test"]['in'].shape)
			deb.prints(self.image["test"]['label'].shape)

	def create_load(self):

		self.patches_list={'train':{},'test':{}}
		#self.patches_list['train']['ims']=glob.glob(self.path['train']['in']+'*.npy')
		#self.patches_list['train']['label']=glob.glob(self.path['train']['label']+'*.npy')

		#self.patches_list['test']['ims']=glob.glob(self.path['test']['in']+'*.npy')
		#self.patches_list['test']['label']=glob.glob(self.path['test']['label']+'*.npy')
		self.patches['test']['label'],self.patches_list['test']['label']=self.folder_load(self.path['test']['label'])
		
		self.patches['train']['in'],self.patches_list['train']['ims']=self.folder_load(self.path['train']['in'])
		self.patches['train']['label'],self.patches_list['train']['label']=self.folder_load(self.path['train']['label'])
		self.patches['test']['in'],self.patches_list['test']['ims']=self.folder_load(self.path['test']['in'])
		deb.prints(self.patches['train']['in'].shape)
		deb.prints(self.patches['test']['in'].shape)
		deb.prints(self.patches['train']['label'].shape)
		self.dataset=None
		unique=np.unique(self.patches['train']['label'])
		deb.prints(unique)
		self.dataset='seq1'

		self.class_n=unique.shape[0] #10 plus background
		print("Switching to one hot")
		self.patches['train']['label']=self.batch_label_to_one_hot(self.patches['train']['label'])
		self.patches['test']['label']=self.batch_label_to_one_hot(self.patches['test']['label'])

		self.patches['train']['in']=self.patches['train']['in'].astype(np.float32)
		self.patches['test']['in']=self.patches['test']['in'].astype(np.float32)

		self.patches['train']['label']=self.patches['train']['label'].astype(np.int8)
		self.patches['test']['label']=self.patches['test']['label'].astype(np.int8)
		
		deb.prints(len(self.patches_list['test']['label']))
		deb.prints(len(self.patches_list['test']['ims']))
		deb.prints(self.patches['train']['in'].shape)
		deb.prints(self.patches['train']['in'].dtype)
		
		deb.prints(self.patches['train']['label'].shape)
		deb.prints(self.patches['test']['label'].shape)
		unique,count = np.unique(self.patches['test']['label'],return_counts=True)
		print_pixel_count = True
		if print_pixel_count == True:
			for t_step in range(self.patches['test']['label'].shape[1]):
				deb.prints(t_step)
				deb.prints(np.unique(self.patches['train']['label'].argmax(axis=-1)[:,t_step],return_counts=True))
			print("Test label unique: ",unique,count)

			for t_step in range(self.patches['test']['label'].shape[1]):
				deb.prints(t_step)
				deb.prints(np.unique(self.patches['test']['label'].argmax(axis=-1)[:,t_step],return_counts=True))
			#pdb.set_trace()
		
		self.patches['train']['n']=self.patches['train']['in'].shape[0]
		self.patches['train']['idx']=range(self.patches['train']['n'])
		np.save('labels_beginning.npy',self.patches['test']['label'])

	def batch_label_to_one_hot(self,im):
		im_one_hot=np.zeros((im.shape[0],im.shape[1],im.shape[2],im.shape[3],self.class_n))
		print(im_one_hot.shape)
		print(im.shape)
		for clss in range(0,self.class_n):
			im_one_hot[:,:,:,:,clss][im[:,:,:,:]==clss]=1
		return im_one_hot

	def folder_load(self,folder_path):
		paths=glob.glob(folder_path+'*.npy')
		files=[]
		deb.prints(len(paths))
		for path in paths:
			#print(path)
			files.append(np.load(path))
		return np.asarray(files),paths
	def subset_create(self, path,patch_step):
		image = self.image_load(path)
		image['label_rgb']=image['label'].copy()
		image['label'] = self.label2idx(image['label'])
		patches = self.patches_extract(image,patch_step)
		return image, patches

	def image_load(self, path):
		image = {}
		image['in'] = cv2.imread(path['in'], -1)
		image['label'] = np.expand_dims(cv2.imread(path['label'], 0), axis=2)
		count,unique=np.unique(image['label'],return_counts=True)
		print("label count,unique",count,unique)
		image['label_rgb']=cv2.imread(path['label'], -1)
		return image

	def patches_extract(self, image, patch_step):

		patches = {}
		patches['in'],_ = self.view_as_windows_multichannel(
			image['in'], (self.patch_len, self.patch_len, self.channel_n), step=patch_step)
		patches['label'],patches['label_partitioned_shape'] = self.view_as_windows_multichannel(
			image['label'], (self.patch_len, self.patch_len, 1), step=patch_step)

		# ===================== Switch labels to one-hot ===============#

		if self.debug >= 2:
			deb.prints(patches['label'].shape)

		if flag['label_one_hot']:

			# Get the vectorized integer label
			patches['label_h'] = np.reshape(
				patches['label'], (patches['label'].shape[0], patches['label'].shape[1]*patches['label'].shape[2]))
			deb.prints(patches['label_h'].shape)

			# Init the one-hot vectorized label
			patches['label_h2'] = np.zeros(
				(patches['label_h'].shape[0], patches['label_h'].shape[1], self.class_n))

			# Get the one-hot vectorized label
			for sample_idx in range(0, patches['label_h'].shape[0]):
				for loc_idx in range(0, patches['label_h'].shape[1]):
					patches['label_h2'][sample_idx, loc_idx,
										patches['label_h'][sample_idx][loc_idx]] = 1

			# Get the image one-hot labels
			patches['label'] = np.reshape(patches['label_h2'], (patches['label_h2'].shape[0],
																patches['label'].shape[1], patches['label'].shape[2], self.class_n))

			if self.debug >= 2:
				deb.prints(patches['label_h2'].shape)

		# ============== End switch labels to one-hot =============#
		if self.debug:
			deb.prints(patches['label'].shape)
			deb.prints(patches['in'].shape)

		return patches

	def label2idx(self, image_label):
		unique = np.unique(image_label)
		idxs = np.array(range(0, unique.shape[0]))
		for val, idx in zip(unique, idxs):
			image_label[image_label == val] = idx
		return image_label

	def view_as_windows_multichannel(self, arr_in, window_shape, step=1):
		out = np.squeeze(view_as_windows(arr_in, window_shape, step=step))
		partitioned_shape=out.shape

		deb.prints(out.shape)
		out = np.reshape(out, (out.shape[0] * out.shape[1],) + out.shape[2::])
		return out,partitioned_shape

#=============== METRICS CALCULATION ====================#
	def ims_flatten(self,ims):
		return np.reshape(ims,(np.prod(ims.shape[0:-1]),ims.shape[-1]))

	def average_acc(self,y_pred,y_true):
		correct_per_class=np.zeros(self.class_n)
		correct_all=y_pred.argmax(axis=1)[y_pred.argmax(axis=1)==y_true.argmax(axis=1)]
		for clss in range(0,self.class_n):
			correct_per_class[clss]=correct_all[correct_all==clss].shape[0]
		if self.debug>=1:
			deb.prints(correct_per_class)

		pred_unique,pred_class_count=np.unique(y_pred.argmax(axis=1),return_counts=True)
		deb.prints(pred_class_count)
		deb.prints(pred_unique+1)


		unique,per_class_count=np.unique(y_true.argmax(axis=1),return_counts=True)
		deb.prints(per_class_count)
		deb.prints(unique+1)
		per_class_count_all=np.zeros(self.class_n)
		for clss,count in zip(unique,per_class_count):
			per_class_count_all[clss]=count
		per_class_acc=np.divide(correct_per_class[1:].astype('float32'),per_class_count_all[1:].astype('float32'))
		average_acc=np.average(per_class_acc)
		return average_acc,per_class_acc
	def flattened_to_im(self,data_h,im_shape):
		return np.reshape(data_h,im_shape)

	def probabilities_to_one_hot(self,vals):
		out=np.zeros_like(vals)
		out[np.arange(len(vals)), vals.argmax(1)] = 1
		return out
	def assert_equal(self,val1,val2):
		return np.equal(val1,val2)
	def int2one_hot(self,x,class_n):
		out = np.zeros((x.shape[0], class_n))
		out[np.arange(x.shape[0]),x] = 1
		return out

	def label_bcknd_from_last_eliminate(self,label):
		out=np.zeros_like(label)
		label_shape=label.shape
		label=np.reshape(label,(label.shape[0],-1))
		out=label[label!=label_shape[-1]-1,:] # label whose value is the last (bcknd)
		out=np.reshape(out,((out.shape[0],)+label_shape[1:]))
		return out

	def metrics_get(self,data,ignore_bcknd=True,debug=2): #requires batch['prediction'],batch['label']
		class_n=data['prediction'].shape[-1]
		#print("label unque at start of metrics_get",
		#	np.unique(data['label'].argmax(axis=4),return_counts=True))
		

		#data['label'][data['label'][:,],:,:,:,:]
		#data['label_copy']=data['label_copy'][:,:,:,:,:-1] # Eliminate bcknd dimension after having eliminated bcknd samples
		
		#print("label_copy unque at start of metrics_get",
	#		np.unique(data['label_copy'].argmax(axis=4),return_counts=True))
		deb.prints(data['prediction'].shape,debug,2)
		deb.prints(data['label'].shape,debug,2)
		#deb.prints(data['label_copy'].shape,debug,2)

		# ==========================IMGS FLATTEN ==========================================#
		data['prediction_h'] = self.ims_flatten(data['prediction'])
		deb.prints(data['prediction_h'].shape,debug,2)

		data['prediction_h']=self.probabilities_to_one_hot(data['prediction_h'])
		deb.prints(data['prediction_h'].shape,debug,2)
				
		data['label_h'] = self.ims_flatten(data['label']) #(self.batch['test']['size']*self.patch_len*self.patch_len,self.class_n
		deb.prints(data['label'].shape,debug,2)
		
		data['label_h_int']=data['label_h'].argmax(axis=1)
		data['prediction_h_int']=data['prediction_h'].argmax(axis=1)

		data['prediction_h_int']=data['prediction_h_int'][data['label_h_int']!=class_n]
		data['label_h_int']=data['label_h_int'][data['label_h_int']!=class_n]

		data['label_h'] = self.int2one_hot(data['label_h_int'],class_n)
		data['prediction_h'] = self.int2one_hot(data['prediction_h_int'],class_n)
		print("After passing to int then to onehot")
		deb.prints(data['label_h'].shape,debug,2)
		deb.prints(data['prediction_h'].shape,debug,2)
				
		ignore_bcknd=False
		if ignore_bcknd==True:
			data['prediction_h']=data['prediction_h'][:,1:]
			data['label_h']=data['label_h'][:,1:]

			if debug>0:
				deb.prints(data['label_h'].shape)
				deb.prints(data['prediction_h'].shape)
			#indices_to_keep=data['prediction_h']
			#data['prediction_h']=data['prediction_h'][:,data['prediction_h']!=0]
			data['prediction_h']=data['prediction_h'][~np.all(data['label_h'] == 0, axis=1)]
			data['label_h']=data['label_h'][~np.all(data['label_h'] == 0, axis=1)]
			
			#for row in range(0,data['label_h'].shape[0]):
			#	if np.sum(data['label_h'][row,:])==0:
			#		np.delete(data['label_h'],row,0)
			#		np.delete(data['prediction_h'],row,0)


		if debug>=1: 
			deb.prints(data['prediction_h'].dtype)
			deb.prints(data['label_h'].dtype)
			deb.prints(data['prediction_h'].shape)
			deb.prints(data['label_h'].shape)
			deb.prints(data['label_h'][0])
			deb.prints(data['prediction_h'][0])

		#============= TEST UNIQUE PRINTING==================#
		unique,count=np.unique(data['label_h'].argmax(axis=1),return_counts=True)
		print("Metric real unique+1,count",unique+1,count)
		unique,count=np.unique(data['prediction_h'].argmax(axis=1),return_counts=True)
		print("Metric prediction unique+1,count",unique+1,count)
		
		#========================METRICS GET================================================#
		metrics={}
		metrics['f1_score']=f1_score(data['label_h'],data['prediction_h'],average='macro')
		metrics['f1_score_weighted']=f1_score(data['label_h'],data['prediction_h'],average='weighted')
		
		metrics['overall_acc']=accuracy_score(data['label_h'],data['prediction_h'])
		metrics['confusion_matrix']=confusion_matrix(data['label_h'].argmax(axis=1),data['prediction_h'].argmax(axis=1))
		metrics['per_class_acc']=(metrics['confusion_matrix'].astype('float') / metrics['confusion_matrix'].sum(axis=1)[:, np.newaxis]).diagonal()
		
		metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])

		
		#=====================IMG RECONSTRUCT============================================#
		if False==True:
			data_label_reconstructed=self.flattened_to_im(data['label_h'],data['label'].shape)
			data_prediction_reconstructed=self.flattened_to_im(data['prediction_h'],data['label'].shape)
		
			deb.prints(data_label_reconstructed.shape)
			np.testing.assert_almost_equal(data['label'],data_label_reconstructed)
			print("Is label reconstructed equal to original",np.array_equal(data['label'],data_label_reconstructed))
			print("Is prediction reconstructed equal to original",np.array_equal(data['prediction'].argmax(axis=3),data_prediction_reconstructed.argmax(axis=3)))

		if self.debug>=2: print(metrics['per_class_acc'])

		return metrics

	def metrics_write_to_txt(self,metrics,loss,epoch=0,path=None):
		#with open(self.report['best']['text_path'], "w") as text_file:
		#    text_file.write("Overall_acc,average_acc,f1_score: {0},{1},{2},{3}".format(str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(epoch)))
		#deb.prints(loss)
		#deb.prints(loss[0])
		#deb.prints(loss[1])
		#dataset='campo_verde'
		if self.dataset=='hannover':
			with open(path, "a") as text_file:
				#text_file.write("{0},{1},{2},{3}\n".format(str(epoch),str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score'])))
				text_file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}\n".format(str(epoch),
					str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(metrics['f1_score_weighted']),str(loss[0]),str(loss[1]),
					str(metrics['per_class_acc'][0]),str(metrics['per_class_acc'][1]),str(metrics['per_class_acc'][2]),
					str(metrics['per_class_acc'][3]),str(metrics['per_class_acc'][4]),str(metrics['per_class_acc'][5]),
					str(metrics['per_class_acc'][6]),str(metrics['per_class_acc'][7])))
		elif metrics['per_class_acc'].shape[0]==10:
			with open(path, "a") as text_file:
				#text_file.write("{0},{1},{2},{3}\n".format(str(epoch),str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score'])))
				text_file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}\n".format(str(epoch),
					str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(metrics['f1_score_weighted']),str(loss[0]),str(loss[1]),
					str(metrics['per_class_acc'][0]),str(metrics['per_class_acc'][1]),str(metrics['per_class_acc'][2]),
					str(metrics['per_class_acc'][3]),str(metrics['per_class_acc'][4]),str(metrics['per_class_acc'][5]),
					str(metrics['per_class_acc'][6]),str(metrics['per_class_acc'][7]),str(metrics['per_class_acc'][8]),
					str(metrics['per_class_acc'][9])))
		elif metrics['per_class_acc'].shape[0]==9:
			with open(path, "a") as text_file:
				#text_file.write("{0},{1},{2},{3}\n".format(str(epoch),str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score'])))
				text_file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}\n".format(str(epoch),
					str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(metrics['f1_score_weighted']),str(loss[0]),str(loss[1]),
					str(metrics['per_class_acc'][0]),str(metrics['per_class_acc'][1]),str(metrics['per_class_acc'][2]),
					str(metrics['per_class_acc'][3]),str(metrics['per_class_acc'][4]),str(metrics['per_class_acc'][5]),
					str(metrics['per_class_acc'][6]),str(metrics['per_class_acc'][7]),str(metrics['per_class_acc'][8])))
		elif metrics['per_class_acc'].shape[0]==8:
			with open(path, "a") as text_file:
				#text_file.write("{0},{1},{2},{3}\n".format(str(epoch),str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score'])))
				text_file.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}\n".format(str(epoch),
					str(metrics['overall_acc']),str(metrics['average_acc']),str(metrics['f1_score']),str(metrics['f1_score_weighted']),str(loss[0]),str(loss[1]),
					str(metrics['per_class_acc'][0]),str(metrics['per_class_acc'][1]),str(metrics['per_class_acc'][2]),
					str(metrics['per_class_acc'][3]),str(metrics['per_class_acc'][4]),str(metrics['per_class_acc'][5]),
					str(metrics['per_class_acc'][6]),str(metrics['per_class_acc'][7])))
				
			
	def metrics_per_class_from_im_get(self,name='im_reconstructed_rgb_test_predictionplen64_3.png',folder='../results/reconstructed/',average=None):
		data={}
		metrics={}
		deb.prints(folder+name)
		data['prediction']=cv2.imread(folder+name,0)[0:-30,0:-2]
		data['label']=cv2.imread(folder+'im_reconstructed_rgb_test_labelplen64_3.png',0)[0:-30,0:-2]

		data['prediction']=np.reshape(data['prediction'],-1)
		data['label']=np.reshape(data['label'],-1)
		
		metrics['f1_score_per_class']=f1_score(data['prediction'],data['label'],average=average)
		print(metrics)


# =================== Image reconstruct =======================#

	def im_reconstruct(self,subset='test',mode='prediction'):
		h,w,_=self.image[subset]['label'].shape
		print(self.patches[subset]['label_partitioned_shape'])
		deb.prints(self.patches[subset][mode].shape)
		
		h_blocks,w_blocks,patch_len,_=self.patches[subset]['label_partitioned_shape']

		patches_block=np.reshape(self.patches[subset][mode].argmax(axis=3),(h_blocks,w_blocks,patch_len,patch_len))


		self.im_reconstructed=np.squeeze(np.zeros_like(self.image[subset]['label']))

		h_block_len=int(self.image[subset]['label'].shape[0]/h_blocks)
		w_block_len=int(self.image[subset]['label'].shape[1]/w_blocks)
		
		count=0

		for w_block in range(0,w_blocks):
			for h_block in range(0,h_blocks):
				y=int(h_block*h_block_len)
				x=int(w_block*w_block_len)
				#print(y)
				#print(x)				
				#deb.prints([y:y+self.patch_len])
				self.im_reconstructed[y:y+self.patch_len,x:x+self.patch_len]=patches_block[h_block,w_block,:,:]
				count+=1

		self.im_reconstructed_rgb=self.im_gray_idx_to_rgb(self.im_reconstructed)
		if self.debug>=3: 
			deb.prints(count)
			deb.prints(self.im_reconstructed_rgb.shape)

		cv2.imwrite('../results/reconstructed/im_reconstructed_rgb_'+subset+'_'+mode+self.report['exp_id']+'.png',self.im_reconstructed_rgb.astype(np.uint8))

	def im_gray_idx_to_rgb(self,im):
		out=np.zeros((im.shape+(3,)))
		for chan in range(0,3):
			for clss in range(0,self.class_n):
				out[:,:,chan][im==clss]=np.array(self.im_gray_idx_to_rgb_table[clss][1][chan])
		deb.prints(out.shape)
		out=cv2.cvtColor(out.astype(np.uint8),cv2.COLOR_RGB2BGR)
		return out
	def val_set_get(self,mode='stratified',validation_split=0.2):
		clss_train_unique,clss_train_count=np.unique(self.patches['train']['label'].argmax(axis=4),return_counts=True)
		deb.prints(clss_train_count)
		self.patches['val']={'n':int(self.patches['train']['n']*validation_split)}
		
		#===== CHOOSE VAL IDX
		#mode='stratified'
		if mode=='random':
			self.patches['val']['idx']=np.random.choice(self.patches['train']['idx'],self.patches['val']['n'],replace=False)
			

			self.patches['val']['in']=self.patches['train']['in'][self.patches['val']['idx']]
			self.patches['val']['label']=self.patches['train']['label'][self.patches['val']['idx']]
		
		elif mode=='stratified':
			# self.patches['train']['in'] are the input sequences of images of shape (val_sample_n,t_len,h,w,channel_n)
			# self.patches['train']['label'] is the ground truth of shape (sample_n,t_len,h,w,class_n)
			# self.patches['val']['in'] are the input sequences of images of shape (val_sample_n,t_len,h,w,channel_n)
			# self.patches['val']['label'] is the ground truth of shape (sample_n,t_len,h,w,class_n)

			while True:
				self.patches['val']['idx']=np.random.choice(self.patches['train']['idx'],self.patches['val']['n'],replace=False)
				self.patches['val']['in']=self.patches['train']['in'][self.patches['val']['idx']]
				self.patches['val']['label']=self.patches['train']['label'][self.patches['val']['idx']]
		
				clss_val_unique,clss_val_count=np.unique(self.patches['val']['label'].argmax(axis=4),return_counts=True)

				# If validation set doesn't contain ALL classes from train set, repeat random choice
				if not np.array_equal(clss_train_unique,clss_val_unique):
					deb.prints(clss_train_unique)
					deb.prints(clss_val_unique)
					pass
				else:
					percentages=clss_val_count/clss_train_count
					deb.prints(percentages)

					# Percentage for each class is equal to: (validation sample number)/(train sample number)*100
					# If percentage from any class is larger than 20% repeat random choice
					if np.any(percentages>0.2):
					
						pass
					else:
						# Else keep the validation set
						break
		elif mode=='random_v2':
			while True:

				self.patches['val']['idx']=np.random.choice(self.patches['train']['idx'],self.patches['val']['n'],replace=False)				

				self.patches['val']['in']=self.patches['train']['in'][self.patches['val']['idx']]
				self.patches['val']['label']=self.patches['train']['label'][self.patches['val']['idx']]
				clss_val_unique,clss_val_count=np.unique(self.patches['val']['label'].argmax(axis=3),return_counts=True)
						
				deb.prints(clss_train_unique)
				deb.prints(clss_val_unique)

				deb.prints(clss_train_count)
				deb.prints(clss_val_count)

				clss_train_count_in_val=clss_train_count[np.isin(clss_train_unique,clss_val_unique)]
				percentages=clss_val_count/clss_train_count_in_val
				deb.prints(percentages)
				#if np.any(percentages<0.1) or np.any(percentages>0.3):
				if np.any(percentages>0.26):
					pass
				else:
					break				

		deb.prints(self.patches['val']['idx'].shape)

		
		deb.prints(self.patches['val']['in'].shape)
		#deb.prints(data.patches['val']['label'].shape)
		
		self.patches['train']['in']=np.delete(self.patches['train']['in'],self.patches['val']['idx'],axis=0)
		self.patches['train']['label']=np.delete(self.patches['train']['label'],self.patches['val']['idx'],axis=0)
		#deb.prints(data.patches['train']['in'].shape)
		#deb.prints(data.patches['train']['label'].shape)
	def semantic_balance(self,samples_per_class=500): # samples mean sequence of patches. Keep
		print("data.semantic_balance")
		
		# Count test
		patch_count=np.zeros(self.class_n)

		for clss in range(self.class_n):
			patch_count[clss]=np.count_nonzero(np.isin(self.patches['test']['label'].argmax(axis=4),clss).sum(axis=(1,2,3)))
		deb.prints(patch_count.shape)
		print("Test",patch_count)
		
		# Count train
		patch_count=np.zeros(self.class_n)

		for clss in range(self.class_n):
			patch_count[clss]=np.count_nonzero(np.isin(self.patches['train']['label'].argmax(axis=4),clss).sum(axis=(1,2,3)))
		deb.prints(patch_count.shape)
		print("Train",patch_count)
		
		# Start balancing
		balance={}
		balance["out_n"]=(self.class_n-1)*samples_per_class
		balance["out_in"]=np.zeros((balance["out_n"],) + self.patches["train"]["in"].shape[1::])

		balance["out_labels"]=np.zeros((balance["out_n"],) + self.patches["train"]["label"].shape[1::])

		label_int=self.patches['train']['label'].argmax(axis=4)
		labels_flat=np.reshape(label_int,(label_int.shape[0],np.prod(label_int.shape[1:])))
		k=0
		for clss in range(1,self.class_n):
			if patch_count[clss]==0:
				continue
			print(labels_flat.shape)
			print(clss)
			#print((np.count_nonzero(np.isin(labels_flat,clss))>0).shape)
			idxs=np.any(labels_flat==clss,axis=1)
			print(idxs.shape,idxs.dtype)
			#labels_flat[np.count_nonzero(np.isin(labels_flat,clss))>0]

			balance["in"]=self.patches['train']['in'][idxs]
			balance["label"]=self.patches['train']['label'][idxs]


			print(clss)
			if balance["label"].shape[0]>samples_per_class:
				replace=False
				index_squeezed=range(balance["label"].shape[0])
				index_squeezed = np.random.choice(index_squeezed, samples_per_class, replace=replace)
				#print(idxs.shape,index_squeezed.shape)
				balance["out_labels"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["label"][index_squeezed]
				balance["out_in"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["in"][index_squeezed]
			else:

				augmented_manipulations=True
				if augmented_manipulations==True:
					augmented_data = balance["in"]
					augmented_labels = balance["label"]

					cont_transf = 0
					for i in range(int(samples_per_class/balance["label"].shape[0] - 1)):                
						augmented_data_temp = balance["in"]
						augmented_label_temp = balance["label"]
						
						if cont_transf == 0:
							augmented_data_temp = np.rot90(augmented_data_temp,1,(2,3))
							augmented_label_temp = np.rot90(augmented_label_temp,1,(2,3))
						
						elif cont_transf == 1:
							augmented_data_temp = np.rot90(augmented_data_temp,2,(2,3))
							augmented_label_temp = np.rot90(augmented_label_temp,2,(2,3))

						elif cont_transf == 2:
							augmented_data_temp = np.flip(augmented_data_temp,2)
							augmented_label_temp = np.flip(augmented_label_temp,2)
							
						elif cont_transf == 3:
							augmented_data_temp = np.flip(augmented_data_temp,3)
							augmented_label_temp = np.flip(augmented_label_temp,3)
						
						elif cont_transf == 4:
							augmented_data_temp = np.rot90(augmented_data_temp,3,(2,3))
							augmented_label_temp = np.rot90(augmented_label_temp,3,(2,3))
							
						elif cont_transf == 5:
							augmented_data_temp = augmented_data_temp
							augmented_label_temp = augmented_label_temp
							
						cont_transf+=1
						if cont_transf==6:
							cont_transf = 0
						print(augmented_data.shape,augmented_data_temp.shape)				
						augmented_data = np.vstack((augmented_data,augmented_data_temp))
						augmented_labels = np.vstack((augmented_labels,augmented_label_temp))
						
		#            augmented_labels_temp = np.tile(clss_labels,samples_per_class/num_samples )
					#print(augmented_data.shape)
					#print(augmented_labels.shape)
					index = range(augmented_data.shape[0])
					index = np.random.choice(index, samples_per_class, replace=True)
					balance["out_labels"][k*samples_per_class:k*samples_per_class + samples_per_class] = augmented_labels[index]
					balance["out_in"][k*samples_per_class:k*samples_per_class + samples_per_class] = augmented_data[index]
				else:
					replace=True
					index = range(balance["label"].shape[0])
					index = np.random.choice(index, samples_per_class, replace=replace)
					balance["out_labels"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["label"][index]
					balance["out_in"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["in"][index]

			k+=1
		
		idx = np.random.permutation(balance["out_labels"].shape[0])
		self.patches['train']['in'] = balance["out_in"][idx]
		self.patches['train']['label'] = balance["out_labels"][idx]

		deb.prints(np.unique(self.patches['train']['label'],return_counts=True))
		# Replicate
		#balance={}
		#for clss in range(1,self.class_n):
		#	balance["data"]=data["train"]["in"][]
			

		#train_flat=np.reshape(self.patches['train']['label'],(self.patches['train']['label'].shape[0],np.prod(self.patches['train']['label'].shape[1:])))
		#deb.prints(train_flat.shape)

		#unique,counts=np.unique(train_flat,axis=1,return_counts=True)
		#print(unique,counts)
# ========== NetModel object implements model graph definition, train/testing, early stopping ================ #

class NetModel(NetObject):
	def __init__(self, batch_size_train=32, batch_size_test=200, epochs=30000, 
		patience=10, eval_mode='metrics', val_set=True,
		model_type='DenseNet', time_measure=False, *args, **kwargs):

		super().__init__(*args, **kwargs)
		if self.debug >= 1:
			print("Initializing Model instance")
		self.val_set=val_set
		self.metrics = {'train': {}, 'test': {}, 'val':{}}
		self.batch = {'train': {}, 'test': {}, 'val':{}}
		self.batch['train']['size'] = batch_size_train
		self.batch['test']['size'] = batch_size_test
		self.batch['val']['size'] = batch_size_test
		
		self.eval_mode = eval_mode
		self.epochs = epochs
		self.early_stop={'best':0,
					'count':0,
					'signal':False,
					'patience':patience}
		self.model_type=model_type
		with open(self.report['best']['text_history_path'], "w") as text_file:
			text_file.write("epoch,oa,aa,f1,class_acc\n")

		with open(self.report['val']['history_path'], "w") as text_file:
			text_file.write("epoch,oa,aa,f1,class_acc\n")

		self.model_save=True
		self.time_measure=time_measure
		self.mp=load_obj('model_params')
	def transition_down(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		
		return pipe

	def dense_block(self, pipe, filters):
		pipe = Conv2D(filters, (3, 3), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		return pipe

	def transition_up(self, pipe, filters):
		pipe = Conv2DTranspose(filters, (3, 3), strides=(
			2, 2), padding='same')(pipe)
		pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		pipe = Activation('relu')(pipe)
		#pipe = Dropout(0.2)(pipe)
		#pipe = Conv2D(filters, (1, 1), padding='same')(pipe)
		#pipe = keras.layers.BatchNormalization(axis=3)(pipe)
		#pipe = Activation('relu')(pipe)
		return pipe

	def concatenate_transition_up(self, pipe1, pipe2, filters):
		pipe = keras.layers.concatenate([pipe1, pipe2], axis=3)
		pipe = self.transition_up(pipe, filters)
		return pipe

	def build(self):
		in_im = Input(shape=(self.t_len,self.patch_len, self.patch_len, self.channel_n))
		filters = 64

		#x = keras.layers.Permute((1,2,0,3))(in_im)
		x = keras.layers.Permute((2,3,1,4))(in_im)
		
		x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
		#pipe = {'fwd': [], 'bckwd': []}
		c = {'init_up': 0, 'up': 0}
		pipe=[]

		# ================== Transition Down ============================ #
		pipe.append(self.transition_down(x, filters))  # 0 16x16
		pipe.append(self.transition_down(pipe[-1], filters*2))  # 1 8x8
		pipe.append(self.transition_down(pipe[-1], filters*4))  # 2 4x4
		pipe.append(self.transition_down(pipe[-1], filters*8))  # 2 4x4
		c['down']=len(pipe)-1 # Last down-layer idx
		
		# =============== Dense block; no transition ================ #
		#pipe.append(self.dense_block(pipe[-1], filters*16))  # 3 4x4

		# =================== Transition Up ============================= #
		c['up']=c['down'] # First up-layer idx 
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*8))  # 4 8x8
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*4))  # 4 8x8
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*2))  # 5
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters))  # 6

		out = Conv2D(self.class_n, (1, 1), activation='softmax',
					 padding='same')(pipe[-1])
		self.graph = Model(in_im, out)
		print(self.graph.summary())

	def build(self):
		in_im = Input(shape=(self.t_len,self.patch_len, self.patch_len, self.channel_n))
		filters = 64

		#x = keras.layers.Permute((1,2,0,3))(in_im)
		x = keras.layers.Permute((2,3,1,4))(in_im)
		
		x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
		#pipe = {'fwd': [], 'bckwd': []}
		

		x = Conv2D(48, (3, 3), activation='relu',
					 padding='same')(x)



		c = {'init_up': 0, 'up': 0}
		pipe=[]

		# ================== Transition Down ============================ #
		pipe.append(self.transition_down(x, filters))  # 0 16x16
		pipe.append(self.transition_down(pipe[-1], filters*2))  # 1 8x8
		#pipe.append(self.transition_down(pipe[-1], filters*4))  # 2 4x4
		c['down']=len(pipe)-1 # Last down-layer idx
		
		# =============== Dense block; no transition ================ #
		#pipe.append(self.dense_block(pipe[-1], filters*16))  # 3 4x4

		# =================== Transition Up ============================= #
		c['up']=c['down'] # First up-layer idx 
		#pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*4))  # 4 8x8
		#c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters*2))  # 5
		c['up']-=1
		pipe.append(self.concatenate_transition_up(pipe[-1], pipe[c['up']], filters))  # 6

		out = Conv2D(self.class_n, (1, 1), activation='softmax',
					 padding='same')(pipe[-1])

		self.graph = Model(in_im, out)
		print(self.graph.summary())

	def build(self):
		deb.prints(self.t_len)
		in_im = Input(shape=(self.t_len,self.patch_len, self.patch_len, self.channel_n))
		weight_decay=1E-4
		def dilated_layer(x,filter_size,dilation_rate=1, kernel_size=3):
			x = TimeDistributed(Conv2D(filter_size, kernel_size, padding='same',
				dilation_rate=(dilation_rate, dilation_rate)))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			return x		
		def transpose_layer(x,filter_size,dilation_rate=1, 
			kernel_size=3, strides=(2,2)):
			x = TimeDistributed(Conv2DTranspose(filter_size, 
				kernel_size, strides=strides, padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			return x		
		def im_pooling_layer(x,filter_size):
			pooling=True
			shape_before=tf.shape(x)
			print("im pooling")
			deb.prints(K.int_shape(x))
			if pooling==True:
				mode=2
				if mode==1:
					x=TimeDistributed(GlobalAveragePooling2D())(x)
					deb.prints(K.int_shape(x))
					x=K.expand_dims(K.expand_dims(x,2),2)
				elif mode==2:
					x=TimeDistributed(AveragePooling2D((32,32)))(x)
					deb.prints(K.int_shape(x))
					
			deb.prints(K.int_shape(x))
			#x=dilated_layer(x,filter_size,1,kernel_size=1)
			deb.prints(K.int_shape(x))

			if pooling==True:
				x = TimeDistributed(Lambda(lambda y: K.tf.image.resize_bilinear(y,size=(32,32))))(x)
#				x = TimeDistributed(UpSampling2D(
#					size=(self.patch_len,self.patch_len)))(x)
			deb.prints(K.int_shape(x))
			print("end im pooling")
			# x=TimeDistributed(Lambda(
			# 	lambda y: tf.compat.v1.image.resize(
			# 		y, shape_before[2:4],
			# 		method='bilinear',align_corners=True)))(x)
			return x
		def spatial_pyramid_pooling(in_im,filter_size,
			max_rate=8,global_average_pooling=False):
			x=[]
			if max_rate>=1:
				x.append(dilated_layer(in_im,filter_size,1))
			if max_rate>=2:
				x.append(dilated_layer(in_im,filter_size,2)) #6
			if max_rate>=4:
				x.append(dilated_layer(in_im,filter_size,4)) #12
			if max_rate>=8:
				x.append(dilated_layer(in_im,filter_size,8)) #18
			if global_average_pooling==True:
				x.append(im_pooling_layer(in_im,filter_size))
			out = keras.layers.concatenate(x, axis=4)
			return out


		if self.model_type=='DenseNet':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			x = keras.layers.Permute((2,3,1,4))(in_im)
			
			x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCN((self.patch_len, self.patch_len, self.t_len*self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_DenseNet':
			# convlstm then densenet

			#x = keras.layers.Permute((2,3,1,4))(in_im)
			x = ConvLSTM2D(32,3,return_sequences=False,padding="same")(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCN((self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM':
			x = ConvLSTM2D(32,3,return_sequences=False,padding="same")(in_im)
			out = Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same')(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='FCN_ConvLSTM':
			x = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(in_im)
			x = ConvLSTM2D(32,3,return_sequences=False,padding="same")(x)
			out = Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same')(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='FCN_ConvLSTM2':
			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				strides=(2, 2)))(in_im)
			e2 = TimeDistributed(Conv2D(32, (3, 3), padding='same',
				strides=(2, 2)))(e1)
			e3 = TimeDistributed(Conv2D(48, (3, 3), padding='same',
				strides=(2, 2)))(e2)

			x = ConvLSTM2D(80,3,return_sequences=False,padding="same")(e3)


			d3 = Conv2DTranspose(48, (3, 3), strides=(
				2, 2), padding='same')(x)
			#d2 = keras.layers.concatenate([d3, e2[:,-1,:,:,:]], axis=3)

			d2 = Conv2DTranspose(32, (3, 3), strides=(
				2, 2), padding='same')(d3)
			#d1 = keras.layers.concatenate([d2, e1[:,-1,:,:,:]], axis=3)
			
			d1 = Conv2DTranspose(16, (3, 3), strides=(
				2, 2), padding='same')(d2)
#			out = keras.layers.concatenate([d1, in_im[:,-1,:,:,:]], axis=3)

			out = Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same')(d1)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='BiConvLSTM_DenseNet':

			# convlstm then densenet

			#x = keras.layers.Permute((2,3,1,4))(in_im)
			x = Bidirectional(ConvLSTM2D(32,3,return_sequences=False,
				padding="same"),merge_mode='concat')(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCN((self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq':
			x = ConvLSTM2D(256,3,return_sequences=True,padding="same")(in_im)
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)
			self.graph = Model(in_im, x)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq_bi':
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(in_im)
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)						 
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq_bi_60x2':
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(in_im)
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			x = Activation('relu')(x)						 
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='FCN_ConvLSTM_seq2seq_bi':
			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				strides=(2, 2)))(in_im)
			e2 = TimeDistributed(Conv2D(32, (3, 3), padding='same',
				strides=(2, 2)))(e1)
			e3 = TimeDistributed(Conv2D(48, (3, 3), padding='same',
				strides=(2, 2)))(e2)

			x = Bidirectional(ConvLSTM2D(80,3,return_sequences=True,
				padding="same"),merge_mode='concat')(e3)


			d3 = TimeDistributed(Conv2DTranspose(48, (3, 3), strides=(
				2, 2), padding='same'))(x)
			#d2 = keras.layers.concatenate([d3, e2[:,-1,:,:,:]], axis=3)

			d2 = TimeDistributed(Conv2DTranspose(32, (3, 3), strides=(
				2, 2), padding='same'))(d3)
			#d1 = keras.layers.concatenate([d2, e1[:,-1,:,:,:]], axis=3)
			
			d1 = TimeDistributed(Conv2DTranspose(16, (3, 3), strides=(
				2, 2), padding='same'))(d2)
#			out = keras.layers.concatenate([d1, in_im[:,-1,:,:,:]], axis=3)

			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
						 padding='same'))(d1)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='DenseNetTimeDistributed':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=16, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=60)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='DenseNetTimeDistributed_128x2':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=self.mp['dense']['nb_dense_block'], growth_rate=self.mp['dense']['growth_rate'], dropout_rate=0.2,
							nb_layers_per_block=self.mp['dense']['nb_layers_per_block'], upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=128)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='DenseNetTimeDistributed_128x2_inconv':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			x=dilated_layer(in_im,16)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=2, growth_rate=64, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=x,
							recurrent_filters=128)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='DenseNetTimeDistributed_128x2_3blocks':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributed((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=3, growth_rate=32, dropout_rate=0.2,
							nb_layers_per_block=2, upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=128)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='pyramid_dilated':

			d1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(2, 2)))(in_im)
			d4 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(4, 4)))(in_im)
			d8 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(8, 8)))(in_im)

			x = keras.layers.concatenate([d1, d4, d8], axis=4)

			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='pyramid_dilated_bconvlstm':

			d1 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(2, 2)))(in_im)
			d1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(d1)
			d1 = Activation('relu')(d1)
			d4 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(4, 4)))(in_im)
			d4 = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(d4)
			d4 = Activation('relu')(d4)
			d8 = TimeDistributed(Conv2D(16, (3, 3), padding='same',
				dilation_rate=(8, 8)))(in_im)
			d8 = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(d8)
			d8 = Activation('relu')(d8)
			pdc = keras.layers.concatenate([d1, d4, d8], axis=4)
			r1 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
							padding="same"))(pdc)
			r2 = Bidirectional(ConvLSTM2D(64,3,return_sequences=True,
							padding="same",dilation_rate=(2, 2)))(pdc)
			x = keras.layers.concatenate([r1, r2], axis=4)



			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='bdepthconvlstm':

			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(in_im)
			e1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(e1)
			e1 = Activation('relu')(e1)
		elif self.model_type=='deeplabv3':

			e1 = TimeDistributed(Conv2D(16, (3, 3), padding='same'))(in_im)
			e1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(e1)
			e1 = Activation('relu')(e1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)

			e1 = TimeDistributed(Conv2D(32, (3, 3), padding='same'))(e1)
			e1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(e1)
			e1 = Activation('relu')(e1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)



			dil1 = TimeDistributed(Conv2D(64, (3, 3), padding='same',
				dilation_rate=(2,2)))(e1)
			dil1 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(dil1)
			dil1 = Activation('relu')(dil1)


			dil2 = TimeDistributed(Conv2D(64, (3, 3), padding='same',
				dilation_rate=(4,4)))(e1)
			dil2 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(dil2)
			dil2 = Activation('relu')(dil2)

			dil3 = TimeDistributed(Conv2D(64, (3, 3), padding='same',
				dilation_rate=(8,8)))(e1)
			dil3 = BatchNormalization(gamma_regularizer=l2(weight_decay),
												beta_regularizer=l2(weight_decay))(dil3)
			dil3 = Activation('relu')(dil3)


			pdc = keras.layers.concatenate([dil1, dil2, dil3], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)


			x = TimeDistributed(UpSampling2D(size=(4, 4)))(x)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
		elif self.model_type=='deeplab_rs':



			d1=dilated_layer(in_im,16,1)
			d2=dilated_layer(in_im,16,2)
			d4=dilated_layer(in_im,16,4)
			d8=dilated_layer(in_im,16,8)

			pdc = keras.layers.concatenate([d1, d2, d4, d8], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,32,1)
			d2=dilated_layer(pdc,32,2)
			d4=dilated_layer(pdc,32,4)

			pdc = keras.layers.concatenate([d1, d2, d4], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,64,1)
			d2=dilated_layer(pdc,64,2)
			pdc = keras.layers.concatenate([d1, d2], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)
			x = TimeDistributed(UpSampling2D(size=(4, 4)))(x)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='deeplab_rs_multiscale':

			d1=dilated_layer(in_im,16,1)
			d2=dilated_layer(in_im,16,2)
			d4=dilated_layer(in_im,16,4)
			d8=dilated_layer(in_im,16,8)

			pdc1 = keras.layers.concatenate([d1, d2, d4, d8], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc1)

			d1=dilated_layer(pdc,32,1)
			d2=dilated_layer(pdc,32,2)
			d4=dilated_layer(pdc,32,4)

			pdc2 = keras.layers.concatenate([d1, d2, d4], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc2)

			d1=dilated_layer(pdc,64,1)
			d2=dilated_layer(pdc,64,2)
			pdc = keras.layers.concatenate([d1, d2], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)
	
			# Decoder V3+
			x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			x2 = dilated_layer(pdc2,128,1,kernel_size=1)#Low level features
			x= keras.layers.concatenate([x, x2], axis=4)
			x= dilated_layer(x,64,1,kernel_size=3)
			x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
		elif self.model_type=='deeplabv3plus':

			e1 = dilated_layer(in_im,16,1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)
			p1 = dilated_layer(e1,32,1)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)



			dilAvg = im_pooling_layer(e1,64)
			dil0 = dilated_layer(e1,64,1,kernel_size=1)
			dil1 = dilated_layer(e1,64,2)
			dil2 = dilated_layer(e1,64,4)
			dil3 = dilated_layer(e1,64,8)

			pdc = keras.layers.concatenate([dilAvg, dil0, dil1, dil2, dil3], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)

			# Decoder V3+
			x = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			x2 = dilated_layer(p1,128,1,kernel_size=1)#Low level features
			x= keras.layers.concatenate([x, x2], axis=4)
			x= dilated_layer(x,64,1,kernel_size=3)
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			out = TimeDistributed(UpSampling2D(size=(2, 2)))(x)
			

			self.graph = Model(in_im, out)
		elif self.model_type=='FCN_ConvLSTM_seq2seq_bi_skip':

			#fs=32
			fs=16
			
			p1=dilated_layer(in_im,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			
		if self.model_type=='BUnetConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='Unet4ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = ConvLSTM2D(256,3,return_sequences=True,
					padding="same")(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet3ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet4ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet5ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			#p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet2ConvLSTM':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e2)

			d2 = transpose_layer(x,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='UnetTimeDistributed':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			#x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
			#        padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(e3,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='BAtrousConvLSTM':

			#fs=32
			fs=16
			
			#x=dilated_layer(in_im,fs)
			x=dilated_layer(in_im,fs)
			x=dilated_layer(x,fs)
			x=spatial_pyramid_pooling(x,fs*4,max_rate=8,
				global_average_pooling=False)
			
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(x)

			#out=dilated_layer(x,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='BAtrousGAPConvLSTM':

			#fs=32
			fs=16
			
			#x=dilated_layer(in_im,fs)
			x=dilated_layer(in_im,fs)
			x=dilated_layer(x,fs)
			x=spatial_pyramid_pooling(x,fs*4,max_rate=8,
				global_average_pooling=True)
			
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(x)

			out=dilated_layer(x,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='BUnetAtrousConvLSTM':

			#fs=32
			fs=16
			x=dilated_layer(in_im,fs)
			p1=spatial_pyramid_pooling(x,fs,max_rate=8)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=spatial_pyramid_pooling(e1,fs*2,max_rate=4)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=spatial_pyramid_pooling(e2,fs*4,max_rate=2)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d2 = transpose_layer(d3,fs*4)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d1 = transpose_layer(d2,fs*2)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='BUnetAtrousConvLSTM_v3p':

			fs=16
			x=dilated_layer(in_im,fs)
			d1=dilated_layer(in_im,16,1)
			d2=dilated_layer(in_im,16,2)
			d4=dilated_layer(in_im,16,4)
			d8=dilated_layer(in_im,16,8)

			pdc = keras.layers.concatenate([d1, d2, d4, d8], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,32,1)
			d2=dilated_layer(pdc,32,2)
			d4=dilated_layer(pdc,32,4)

			pdc = keras.layers.concatenate([d1, d2, d4], axis=4)
			pdc = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(pdc)

			d1=dilated_layer(pdc,64,1)
			d2=dilated_layer(pdc,64,2)
			pdc = keras.layers.concatenate([d1, d2], axis=4)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(pdc)
			x = TimeDistributed(UpSampling2D(size=(4, 4)))(x)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='Attention_DenseNetTimeDistributed_128x2':


			#x = keras.layers.Permute((1,2,0,3))(in_im)
			#x = keras.layers.Permute((2,3,1,4))(in_im)
			
			#x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)
			out = DenseNetFCNTimeDistributedAttention((self.t_len, self.patch_len, self.patch_len, self.channel_n), nb_dense_block=self.mp['dense']['nb_dense_block'], growth_rate=self.mp['dense']['growth_rate'], dropout_rate=0.2,
							nb_layers_per_block=self.mp['dense']['nb_layers_per_block'], upsampling_type='deconv', classes=self.class_n, 
							activation='softmax', batchsize=32,input_tensor=in_im,
							recurrent_filters=128)
			self.graph = Model(in_im, out)


		if self.model_type=='fcn_lstm_temouri':


			#fs=32
			fs=64

			#e1=dilated_layer(in_im,fs)			
			e1=dilated_layer(in_im,fs) # 64
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e1)
			e2=dilated_layer(e2,fs*2) # 128
			deb.prints(K.int_shape(e2))
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e2)
			e3=dilated_layer(e3,fs*4) # 256
			deb.prints(K.int_shape(e3))

			e4 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(e3)


			e4=dilated_layer(e4,fs*8) # 512

			d3 = transpose_layer(e4,fs*4) #  256
			deb.prints(K.int_shape(d3))

			d3 = keras.layers.concatenate([d3, e3], axis=4) # 512
			d3=dilated_layer(d3,fs*4) # 256
			d2 = transpose_layer(d3,fs*2) # 128
			deb.prints(K.int_shape(d2))
			d2 = keras.layers.concatenate([d2, e2], axis=4) # 256
			d2=dilated_layer(d2,fs*2) # 128
			d1 = transpose_layer(d2,fs) # 64
			d1 = keras.layers.concatenate([d1, e1], axis=4) # 128
			d1=dilated_layer(d1,fs) # 64 # this would concatenate with the date
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(d1)			
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(x)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='fcn_lstm_temouri2':


			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x=dilated_layer(e3,fs*8)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(out)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())
# ==================================== ATTENTION ATTEMPTS =================================== #
		elif self.model_type=='ConvLSTM_seq2seq_bi_attention':
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			
			# attention

			# shape is (t,h,w,c). We want shape (c,h,w,t)
			# then timedistributed conv. 1x1 applies attention to t
			# then return 
			x = keras.layers.Permute((4,2,3,1))(in_im)
			x = TimeDistributed(Conv2D(self.t_len, (1, 1), activation=None,
						 padding='same'))(x)
			x = Activation('relu')(x)
			x = keras.layers.Permute((4,2,3,1))(x)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(x)

			
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)

#			x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)

			self.graph = Model(in_im, out)
			print(self.graph.summary())

		elif self.model_type=='ConvLSTM_seq2seq_bi_attention2':
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)

			
			# attention

			# shape is (t,h,w,c). We want shape (c,h,w,t)
			# then timedistributed conv. 1x1 applies attention to t
			# then return 
			def attention_weights(x):
				att = keras.layers.Permute((4,2,3,1))(x)
				att = TimeDistributed(Conv2D(self.t_len, (1, 1), activation=None,
							 padding='same'))(att)
				att = Activation('relu')(att)
				att = keras.layers.Permute((4,2,3,1))(att)
				return att
			att = attention_weights(in_im)
			x = multiply([x,att])
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(x)

			
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)

#			x = Reshape((self.patch_len, self.patch_len,self.t_len*self.channel_n), name='predictions')(x)

			self.graph = Model(in_im, out)
			print(self.graph.summary())
		elif self.model_type=='ConvLSTM_seq2seq_bi_SelfAttention':
			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
				padding="same"))(in_im)
#			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation='softmax',
#						 padding='same'))(x)
			x = Reshape((self.t_len,
					32*32*256))(x)
			x = SeqSelfAttention(
				kernel_regularizer=l2(1e-4),
				bias_regularizer=l1(1e-4),
				attention_regularizer_weight=1e-4,
				name='Attention')(x)
			x = Reshape((self.t_len,
					32,32,256))(x)
			x = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
						 padding='same'))(x)
			x = BatchNormalization(gamma_regularizer=l2(weight_decay),
							   beta_regularizer=l2(weight_decay))(x)
			out = Activation('relu')(x)						 
			self.graph = Model(in_im, out)
			print(self.graph.summary())
		if self.model_type=='BUnet4ConvLSTM_SelfAttention':

			print(self.model_type)
			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
					padding="same"),merge_mode='concat')(e3)
			x = Reshape((self.t_len,
					4*4*256))(x)
			x = SeqSelfAttention(
				kernel_regularizer=l2(1e-4),
				bias_regularizer=l1(1e-4),
				attention_regularizer_weight=1e-4,
				name='Attention')(x)
			x = Reshape((self.t_len,
					4,4,256))(x)
			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='Unet4ConvLSTM_SelfAttention':

			print(self.model_type)
			#fs=32
			fs=16

			p1=dilated_layer(in_im,fs)			
			p1=dilated_layer(p1,fs)
			e1 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p1)
			p2=dilated_layer(e1,fs*2)
			e2 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p2)
			p3=dilated_layer(e2,fs*4)
			e3 = TimeDistributed(AveragePooling2D((2, 2), strides=(2, 2)))(p3)

			#x = Bidirectional(ConvLSTM2D(128,3,return_sequences=True,
			#		padding="same"),merge_mode='concat')(e3)
			x = Reshape((self.t_len,
					4*4*fs*4))(e3)
			x = SeqSelfAttention(
				units=256,
				kernel_regularizer=l2(1e-4),
				bias_regularizer=l1(1e-4),
				attention_regularizer_weight=1e-4,
				name='Attention')(x)
			x = Reshape((self.t_len,
					4,4,fs*4))(x)
			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=4)
			d3=dilated_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=4)
			d2=dilated_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=4)
			out=dilated_layer(d1,fs)
			out = TimeDistributed(Conv2D(self.class_n, (1, 1), activation=None,
										padding='same'))(out)
			self.graph = Model(in_im, out)
			print(self.graph.summary())

		if self.model_type=='BUnet4_Standalone':
			print(self.model_type)
			in_shape = (self.t_len,self.patch_len, self.patch_len, self.channel_n)
			#in_im = Input(shape=in_shape)

			#x = Permute((1,2,3,0), input_shape = in_shape)(in_im)
			x = Permute((2,3,4,1), input_shape = in_shape)(in_im)

			#my_permute = lambda y: K.permute_dimensions(y, (None,1,2,3,0))
			#x = Lambda(my_permute)(in_im)
			x = Reshape((self.patch_len,self.patch_len,self.channel_n*self.t_len))(x)
			#fs=32
			def conv_layer(x,filter_size,dilation_rate=1, kernel_size=3):
				x = Conv2D(filter_size, kernel_size, padding='same',
					dilation_rate=(dilation_rate, dilation_rate))(x)
				x = BatchNormalization(gamma_regularizer=l2(weight_decay),
								beta_regularizer=l2(weight_decay))(x)
				x = Activation('relu')(x)
				return x		
			def transpose_layer(x,filter_size,dilation_rate=1, 
				kernel_size=3, strides=(2,2)):
				x = Conv2DTranspose(filter_size, 
					kernel_size, strides=strides, padding='same')(x)
				x = BatchNormalization(gamma_regularizer=l2(weight_decay),
													beta_regularizer=l2(weight_decay))(x)
				x = Activation('relu')(x)
				return x		
			fs=16

			p1=conv_layer(x,fs)			
			p1=conv_layer(p1,fs)
			e1 = AveragePooling2D((2, 2), strides=(2, 2))(p1)
			p2=conv_layer(e1,fs*2)
			e2 = AveragePooling2D((2, 2), strides=(2, 2))(p2)
			p3=conv_layer(e2,fs*4)
			e3 = AveragePooling2D((2, 2), strides=(2, 2))(p3)

			# This replaces convlstm. Check param count and increase filters if lacking
			x = conv_layer(e3, fs*16)

			d3 = transpose_layer(x,fs*4)
			d3 = keras.layers.concatenate([d3, p3], axis=3)
			d3=conv_layer(d3,fs*4)
			d2 = transpose_layer(d3,fs*2)
			d2 = keras.layers.concatenate([d2, p2], axis=3)
			d2=conv_layer(d2,fs*2)
			d1 = transpose_layer(d2,fs)
			d1 = keras.layers.concatenate([d1, p1], axis=3)
			out=conv_layer(d1,fs)
			out = Conv2D(self.class_n*self.t_len, (1, 1), activation=None,
										padding='same')(out)
			#deb.prints(out.output_shape)
			out_shape = (self.patch_len,self.patch_len,self.class_n,self.t_len)
			out = Reshape(out_shape)(out)

			#my_permute = lambda x: K.permute_dimensions(out, (None,3,0,1,2))
			#out = Lambda(my_permute)(x)

			#out = Permute((3,0,1,2), input_shape=out_shape)(out)
			out = Permute((4,1,2,3), input_shape=out_shape)(out)

			self.graph = Model(in_im, out)
			print(self.graph.summary())
		#self.graph = Model(in_im, out)
		print(self.graph.summary(line_length=125))



		with open('model_summary.txt','w') as fh:
			self.graph.summary(line_length=125,print_fn=lambda x: fh.write(x+'\n'))
		#self.graph.summary(print_fn=model_summary_print)
	def compile(self, optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.categorical_accuracy],loss_weights=None):
		#loss_weighted=weighted_categorical_crossentropy(loss_weights)
		loss_weighted=weighted_categorical_crossentropy_ignoring_last_label(loss_weights)
		#sparse_accuracy_ignoring_last_label()
		self.graph.compile(loss=loss_weighted, optimizer=optimizer, metrics=metrics)
		#self.graph.compile(loss=sparse_accuracy_ignoring_last_label, optimizer=optimizer, metrics=metrics)
		#self.graph.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
	def loss_weights_estimate(self,data):
		unique,count=np.unique(data.patches['train']['label'].argmax(axis=4),return_counts=True)
		unique=unique[1:] # No bcknd
		count=count[1:].astype(np.float32)
		weights_from_unique=np.max(count)/count
		deb.prints(weights_from_unique)
		deb.prints(np.max(count))
		deb.prints(count)
		deb.prints(unique)
		self.loss_weights=np.zeros(self.class_n)
		for clss in range(1,self.class_n): # class 0 is bcknd. Leave it in 0
			
			if clss in unique:
				self.loss_weights[clss]=weights_from_unique[unique==clss]
			else:
				self.loss_weights[clss]=0
		deb.prints(self.loss_weights)

		self.loss_weights[1:]=1
		self.loss_weights=self.loss_weights[1:]
		deb.prints(self.loss_weights.shape)
		
	def test(self,data):
		data.patches['train']['batch_n'] = data.patches['train']['in'].shape[0]//self.batch['train']['size']
		data.patches['test']['batch_n'] = data.patches['test']['in'].shape[0]//self.batch['test']['size']

		batch = {'train': {}, 'test': {}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])

		self.metrics['test']['loss'] = np.zeros((1, 2))

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'])
		self.batch_test_stats=True

		for batch_id in range(0, self.batch['test']['n']):
			idx0 = batch_id*self.batch['test']['size']
			idx1 = (batch_id+1)*self.batch['test']['size']

			batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
			batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

			if self.batch_test_stats:
				self.metrics['test']['loss'] += self.graph.test_on_batch(
					batch['test']['in'], batch['test']['label'])		# Accumulated epoch

			data.patches['test']['prediction'][idx0:idx1]=self.graph.predict(batch['test']['in'],batch_size=self.batch['test']['size'])

		#====================METRICS GET================================================#
		deb.prints(data.patches['test']['label'].shape)		
		deb.prints(idx1)
		print("Epoch={}".format(epoch))	
		
		# Average epoch loss
		self.metrics['test']['loss'] /= self.batch['test']['n']
			
		# Get test metrics
		metrics=data.metrics_get(data.patches['test'],debug=1)
		print('oa={}, aa={}, f1={}, f1_wght={}'.format(metrics['overall_acc'],
			metrics['average_acc'],metrics['f1_score'],metrics['f1_score_weighted']))
	def train(self, data):

		# Random shuffle
		##data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'], random_state=0)

		# Normalize
		##data.patches['train']['in'] = normalize(data.patches['train']['in'].astype('float32'))
		##data.patches['test']['in'] = normalize(data.patches['test']['in'].astype('float32'))

		# Computing the number of batches
		data.patches['train']['batch_n'] = data.patches['train']['in'].shape[0]//self.batch['train']['size']
		data.patches['test']['batch_n'] = data.patches['test']['in'].shape[0]//self.batch['test']['size']
		data.patches['val']['batch_n'] = data.patches['val']['in'].shape[0]//self.batch['val']['size']

		deb.prints(data.patches['train']['batch_n'])

		self.train_loop(data)

	def early_stop_check(self,metrics,epoch,most_important='overall_acc'):

		if metrics[most_important]>=self.early_stop['best'] and self.early_stop["signal"]==False:
			self.early_stop['best']=metrics[most_important]
			self.early_stop['count']=0
			print("Best metric updated")
			self.early_stop['best_updated']=True
			#data.im_reconstruct(subset='test',mode='prediction')
		else:
			self.early_stop['best_updated']=False
			self.early_stop['count']+=1
			deb.prints(self.early_stop['count'])
			if self.early_stop["count"]>=self.early_stop["patience"]:
				self.early_stop["signal"]=True

			#else:
				#self.early_stop["signal"]=False
			
			
	def train_loop(self, data):
		print('Start the training')
		cback_tboard = keras.callbacks.TensorBoard(
			log_dir='../summaries/', histogram_freq=0, batch_size=self.batch['train']['size'], write_graph=True, write_grads=False, write_images=False)
		txt={'count':0,'val':{},'test':{}}
		txt['val']={'metrics':[],'epoch':[],'loss':[]}
		txt['test']={'metrics':[],'epoch':[],'loss':[]}
		
		
		#========= VAL INIT


		if self.val_set:
			count,unique=np.unique(data.patches['val']['label'].argmax(axis=4),return_counts=True)
			print("Val label count,unique",count,unique)

		count,unique=np.unique(data.patches['train']['label'].argmax(axis=4),return_counts=True)
		print("Train count,unique",count,unique)
		
		count,unique=np.unique(data.patches['test']['label'].argmax(axis=4),return_counts=True)
		print("Test count,unique",count,unique)
		
		#==================== ESTIMATE BATCH NUMBER===============================#
		batch = {'train': {}, 'test': {}, 'val':{}}
		self.batch['train']['n'] = data.patches['train']['in'].shape[0] // self.batch['train']['size']
		self.batch['test']['n'] = data.patches['test']['in'].shape[0] // self.batch['test']['size']
		self.batch['val']['n'] = data.patches['val']['in'].shape[0] // self.batch['val']['size']

		data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'][:,:,:,:,:-1])
		deb.prints(data.patches['test']['label'].shape)
		deb.prints(self.batch['test']['n'])
		
		self.early_stop["signal"]=False
		#if self.train_mode==

		#data.im_reconstruct(subset='test',mode='label')
		#for epoch in [0,1]:
		init_time=time.time()
		#==============================START TRAIN/TEST LOOP============================#
		for epoch in range(self.epochs):

			idxs=np.random.permutation(data.patches['train']['in'].shape[0])
			data.patches['train']['in']=data.patches['train']['in'][idxs]
			data.patches['train']['label']=data.patches['train']['label'][idxs]
			
			self.metrics['train']['loss'] = np.zeros((1, 2))
			self.metrics['test']['loss'] = np.zeros((1, 2))
			self.metrics['val']['loss'] = np.zeros((1, 2))

			# Random shuffle the data
			##data.patches['train']['in'], data.patches['train']['label'] = shuffle(data.patches['train']['in'], data.patches['train']['label'])

			#=============================TRAIN LOOP=========================================#
			for batch_id in range(0, self.batch['train']['n']):
				
				idx0 = batch_id*self.batch['train']['size']
				idx1 = (batch_id+1)*self.batch['train']['size']

				batch['train']['in'] = data.patches['train']['in'][idx0:idx1]
				batch['train']['label'] = data.patches['train']['label'][idx0:idx1]
				if self.time_measure==True:
					start_time=time.time()
				self.metrics['train']['loss'] += self.graph.train_on_batch(
					batch['train']['in'].astype(np.float32), 
					np.expand_dims(batch['train']['label'].argmax(axis=4),axis=4).astype(np.int8))		# Accumulated epoch
				if self.time_measure==True:
					batch_time=time.time()-start_time
					print(batch_time)
					sys.exit('Batch time:')
			# Average epoch loss
			self.metrics['train']['loss'] /= self.batch['train']['n']

			self.train_predict=True
			#if self.train_predict:



			#================== VAL LOOP=====================#
			if self.val_set:
				data.patches['val']['prediction']=np.zeros_like(data.patches['val']['label'][:,:,:,:,:-1])
				self.batch_test_stats=False

				for batch_id in range(0, self.batch['val']['n']):
					idx0 = batch_id*self.batch['val']['size']
					idx1 = (batch_id+1)*self.batch['val']['size']

					batch['val']['in'] = data.patches['val']['in'][idx0:idx1]
					batch['val']['label'] = data.patches['val']['label'][idx0:idx1]

					if self.batch_test_stats:
						self.metrics['val']['loss'] += self.graph.test_on_batch(
							batch['val']['in'].astype(np.float32), 
							np.expand_dims(batch['val']['label'].argmax(axis=4),axis=4).astype(np.int8))		# Accumulated epoch

					data.patches['val']['prediction'][idx0:idx1]=self.graph.predict(
						batch['val']['in'].astype(np.float32),batch_size=self.batch['val']['size'])
				self.metrics['val']['loss'] /= self.batch['val']['n']

				metrics_val=data.metrics_get(data.patches['val'],debug=2)

				self.early_stop_check(metrics_val,epoch)
				#if epoch==1000 or epoch==700 or epoch==500 or epoch==1200:
				#	self.early_stop['signal']=True
				#else:
				#	self.early_stop['signal']=False
				#if self.early_stop['signal']==True:
				#	self.graph.save('model_'+str(epoch)+'.h5')

				metrics_val['per_class_acc'].setflags(write=1)
				metrics_val['per_class_acc'][np.isnan(metrics_val['per_class_acc'])]=-1
				print(metrics_val['per_class_acc'])
				
				# if epoch % 5 == 0:
				# 	print("Writing val...")
				# 	#print(txt['val']['metrics'])
				# 	for i in range(len(txt['val']['metrics'])):
				# 		data.metrics_write_to_txt(txt['val']['metrics'][i],np.squeeze(txt['val']['loss'][i]),
				# 			txt['val']['epoch'][i],path=self.report['val']['history_path'])
				# 	txt['val']['metrics']=[]
				# 	txt['val']['loss']=[]
				# 	txt['val']['epoch']=[]
				# else:
				# 	txt['val']['metrics'].append(metrics_val)
				# 	txt['val']['loss'].append(self.metrics['val']['loss'])
				# 	txt['val']['epoch'].append(epoch)

			
			#==========================TEST LOOP================================================#
			if self.early_stop['signal']==True:
				self.graph.load_weights('weights_best.h5')
			test_loop_each_epoch=False
			if test_loop_each_epoch==True or self.early_stop['signal']==True:
				data.patches['test']['prediction']=np.zeros_like(data.patches['test']['label'][:,:,:,:,:-1])
				self.batch_test_stats=False

				for batch_id in range(0, self.batch['test']['n']):
					idx0 = batch_id*self.batch['test']['size']
					idx1 = (batch_id+1)*self.batch['test']['size']

					batch['test']['in'] = data.patches['test']['in'][idx0:idx1]
					batch['test']['label'] = data.patches['test']['label'][idx0:idx1]

					if self.batch_test_stats:
						self.metrics['test']['loss'] += self.graph.test_on_batch(
							batch['test']['in'].astype(np.float32), 
							np.expand_dims(batch['test']['label'].argmax(axis=4),axis=4).astype(np.int8))		# Accumulated epoch

					data.patches['test']['prediction'][idx0:idx1]=self.graph.predict(
						batch['test']['in'].astype(np.float32),batch_size=self.batch['test']['size'])


			#====================METRICS GET================================================#
			deb.prints(data.patches['test']['label'].shape)		
			deb.prints(idx1)
			print("Epoch={}".format(epoch))	
			
			if self.batch_test_stats==True:
				# Average epoch loss
				self.metrics['test']['loss'] /= self.batch['test']['n']
			# Get test metrics
			metrics=data.metrics_get(data.patches['test'],debug=1)
			
			if self.early_stop['best_updated']==True:
				if test_loop_each_epoch==True:
					self.early_stop['best_predictions']=data.patches['test']['prediction']
				self.graph.save_weights('weights_best.h5')
				if self.model_save==True:
					self.graph.save('model_best.h5')
				
			print(self.early_stop['signal'])
			if self.early_stop["signal"]==True:
				self.early_stop['best_predictions']=data.patches['test']['prediction']
				print("EARLY STOP EPOCH",epoch,metrics)
				training_time=round(time.time()-init_time,2)
				print("Training time",training_time)
				metadata = "Timestamp:"+ str(round(time.time(),2))+". Model: "+self.model_type+". Training time: "+str(training_time)+"\n"
				print(metadata)
				txt_append("metadata.txt",metadata)
				np.save("prediction.npy",self.early_stop['best_predictions'])
				np.save("labels.npy",data.patches['test']['label'])
				break
				
			# Check early stop and store results if they are the best
			#if epoch % 5 == 0:
			#	print("Writing to file...")
				#for i in range(len(txt['test']['metrics'])):

				#	data.metrics_write_to_txt(txt['test']['metrics'][i],np.squeeze(txt['test']['loss'][i]),
				#		txt['test']['epoch'][i],path=self.report['best']['text_history_path'])
			#	txt['test']['metrics']=[]
			#	txt['test']['loss']=[]
			#	txt['test']['epoch']=[]
				#self.graph.save('my_model.h5')

			else:

				txt['test']['metrics'].append(metrics)
				txt['test']['loss'].append(self.metrics['test']['loss'])
				txt['test']['epoch'].append(epoch)

			#data.metrics_write_to_txt(metrics,np.squeeze(self.metrics['test']['loss']),
			#	epoch,path=self.report['best']['text_history_path'])
			#self.test_metrics_evaluate(data.patches['test'],metrics,epoch)
			#if self.early_stop['signal']==True:
			#	break


			deb.prints(metrics['confusion_matrix'])
			#metrics['average_acc'],metrics['per_class_acc']=self.average_acc(data['prediction_h'],data['label_h'])
			deb.prints(metrics['per_class_acc'])
			if self.val_set:
				deb.prints(metrics_val['per_class_acc'])
			
			print('oa={}, aa={}, f1={}, f1_wght={}'.format(metrics['overall_acc'],
				metrics['average_acc'],metrics['f1_score'],metrics['f1_score_weighted']))
			if self.val_set:
				print('val oa={}, aa={}, f1={}, f1_wght={}'.format(metrics_val['overall_acc'],
					metrics_val['average_acc'],metrics_val['f1_score'],metrics_val['f1_score_weighted']))
			if self.batch_test_stats==True:
				if self.val_set:
				
					print("Loss. Train={}, Val={}, Test={}".format(self.metrics['train']['loss'],
						self.metrics['val']['loss'],self.metrics['test']['loss']))
				else:
					print("Loss. Train={}, Test={}".format(self.metrics['train']['loss'],self.metrics['test']['loss']))
			else:
				print("Train loss",self.metrics['train']['loss'])
			#====================END METRICS GET===========================================#


flag = {"data_create": 2, "label_one_hot": True}
if __name__ == '__main__':
	#
	
	time_measure=False
	#if data.dataset=='seq2':
	#	args.class_n=10

	data = Dataset(patch_len=args.patch_len, patch_step_train=args.patch_step_train,
		patch_step_test=args.patch_step_test,exp_id=args.exp_id,
		path=args.path, t_len=args.t_len, class_n=args.class_n)

	#data.dataset='seq2'

	if flag['data_create']==1:
		data.create()
	elif flag['data_create']==2:
		data.create_load()

	if data.dataset=='seq1' or data.dataset=='seq2':
		args.patience=10
	else:
		args.patience=15
	
	
	val_set=True
	#val_set_mode='stratified'
	val_set_mode='stratified'
	#val_set_mode='random'

	deb.prints(data.patches['train']['label'].shape)

	
	deb.prints(data.patches['train']['label'].shape)
	deb.prints(data.patches['test']['label'].shape)
	
	test_label_unique,test_label_count=np.unique(data.patches['test']['label'].argmax(axis=4),return_counts=True)
	deb.prints(test_label_unique)
	deb.prints(test_label_count)
	train_label_unique,train_label_count=np.unique(data.patches['train']['label'].argmax(axis=4),return_counts=True)
	deb.prints(train_label_unique)
	deb.prints(train_label_count)
	data.label_unique=test_label_unique.copy()
	



	#adam = Adam(lr=0.0001, beta_1=0.9)
	#adam = Adam(lr=0.001, beta_1=0.9)
	
	adam = Adagrad(0.01)
	model = NetModel(epochs=args.epochs, patch_len=args.patch_len,
					 patch_step_train=args.patch_step_train, eval_mode=args.eval_mode,
					 batch_size_train=args.batch_size_train,batch_size_test=args.batch_size_test,
					 patience=args.patience,t_len=args.t_len,class_n=args.class_n,path=args.path,
					 val_set=val_set,model_type=args.model_type, time_measure=time_measure)
	model.class_n=data.class_n-1 # Model is designed without background class
	deb.prints(data.class_n)
	model.build()
	model.class_n+=1 # This is used in loss_weights_estimate, val_set_get, semantic_balance (To-do: Eliminate bcknd class)
	deb.prints(data.patches['train']['label'].shape)

	# === SELECT VALIDATION SET FROM TRAIN SET
	val_set = True # fix this
	if val_set:
		data.val_set_get(val_set_mode,0.15)
		deb.prints(data.patches['val']['label'].shape)
	balancing=True
	if balancing==True:

		
		# If patch balancing
		
		if data.dataset=='seq1' or data.dataset=='seq2':
			#data.semantic_balance(500) #Changed from 1000
			data.semantic_balance(500) #More for seq2seq
			
		else:
			data.semantic_balance(300)
	model.loss_weights_estimate(data)

	model.class_n-=1
	# Label background from 0 to last. 
	deb.prints(data.patches['train']['label'].shape)
	# ===
	def label_bcknd_from_0_to_last(label_one_hot,class_n):		
		print("Changing bcknd from 0 to last...")
		deb.prints(np.unique(label_one_hot.argmax(axis=4),return_counts=True))

		label_h=np.reshape(label_one_hot,(-1,label_one_hot.shape[-1]))
		print("label_h",label_h.shape)
		label_int=label_h.argmax(axis=1)
		label_int[label_int==0]=class_n+1# This number counts the bcknd. So if 3 classes+bcknd=4, class_n=4
		label_int-=1

		out = np.zeros((label_int.shape[0], class_n+1))
		out[np.arange(label_int.shape[0]),label_int]=1
		deb.prints(out.shape)
		out=np.reshape(out,(label_one_hot.shape))

		deb.prints(np.unique(out.argmax(axis=4),return_counts=True))

		return out.astype(np.int8)	

	# def label_bcknd_from_0_to_last(label,class_n):	
	# 	print("Changing bcknd from 0 to last...")
	# 	deb.prints(np.unique(label.argmax(axis=4),return_counts=True))

	# 	out=np.zeros_like(label)
	# 	valid_class_ids=[i for i in range(1,class_n+1)]
	# 	out=label[:,:,:,:,valid_class_ids+[0]]
	# 	deb.prints(np.unique(out.argmax(axis=4),return_counts=True))

	# 	return out


	deb.prints(data.patches['train']['label'][560,:,15,15,:])
	data.patches['train']['label']=label_bcknd_from_0_to_last(
		data.patches['train']['label'],model.class_n)
	deb.prints(data.patches['train']['label'][560,:,15,15,:])

	data.patches['test']['label']=label_bcknd_from_0_to_last(
		data.patches['test']['label'],model.class_n)
	data.patches['val']['label']=label_bcknd_from_0_to_last(
		data.patches['val']['label'],model.class_n)
	data.patches['test']['in']=data.patches['test']['in'].astype(np.float32)
	data.patches['train']['in']=data.patches['train']['in'].astype(np.float32)		
	deb.prints(data.patches['val']['label'].shape)
	#=========== Hannover

	metrics=['accuracy']
	#metrics=['accuracy',fmeasure,categorical_accuracy]
	model.compile(loss='binary_crossentropy',
				  optimizer=adam, metrics=metrics,loss_weights=model.loss_weights)
	model_load=False
	if model_load:
		model=load_model('/home/lvc/Documents/Jorg/sbsr/fcn_model/results/seq2_true_norm/models/model_1000.h5')
		model.test(data)
	
	if args.debug:
		deb.prints(np.unique(data.patches['train']['label']))
		deb.prints(data.patches['train']['label'].shape)
	model.train(data)
