
from model_base import *

# ================================= Implements ConvLSTM ============================================== #
class conv_lstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.remove_sparse_loss=False
		self.model_build()
		
	def model_graph_get(self,data):

		# ConvLSTM Layer (Get last image)
		pipe=self.layer_lstm_get(data,filters=16,kernel=self.kernel,name='convlstm')
		#pipe=self.layer_lstm_multi_get(data,filters=32,kernel=self.kernel,name='multi_convlstm')
		
		if self.debug: deb.prints(pipe.get_shape())

		# Flatten		
		pipe = tf.contrib.layers.flatten(pipe)
		if self.debug: deb.prints(pipe.get_shape())

		# Dense
		pipe = tf.layers.dense(pipe, 256,activation=tf.nn.tanh,name='hidden')

		#pipe = tf.nn.tanh(pipe)
		if self.debug: deb.prints(pipe.get_shape())

		# Dropout
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		
		# Final dense
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())
		return None,pipe

# ================================= Implements BasicLSTM ============================================== #
class lstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):

		# Prints data shape
		if self.debug: deb.prints(data.get_shape())
		
		# Flatten images from each sequence member
		pipe = tf.reshape(data,[-1,self.timesteps,self.patch_len*self.patch_len*self.channels]) 
		if self.debug: deb.prints(pipe.get_shape())
		
		# BasicLSTM layer (Get last image)
		pipe=self.layer_flat_lstm_get(pipe,filters=128,kernel=self.kernel,name='convlstm')
		
		if self.debug: deb.prints(pipe.get_shape())
		
		# Dense
		pipe = tf.layers.dense(pipe, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(pipe.get_shape())
		
		# Dropout
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		
		# Final dense
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())
		return None,pipe

# ================================= Implements U-Net ============================================== #

# Remote: python main.py -mm="ram" --debug=2 -po 27 -ts 5 -tnl 10000000 --batch_size=50 --filters=256 -pl=32 -m="smcnn_unet" -nap=16000
# Local: python main.py -mm="ram" --debug=1 -po 27 -bs=500 --filters=32 -m="smcnn_unet" -pl=32 -nap=16000
class UNet(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=256
	def model_graph_get(self,data):
		pipe = tf.gather(data, int(data.get_shape()[1]) - 1,axis=1)
		pipe = tf.layers.conv2d(pipe, self.filters, self.kernel_size, activation=tf.nn.tanh,padding='same')
		pipe = tf.layers.conv2d(pipe, self.n_classes, self.kernel_size, activation=None,padding='same')
		prediction = tf.argmax(pipe, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), pipe
		return pipe, prediction
	def conv_block_get(self,pipe):
		pipe = tf.layers.conv2d(pipe, self.filters, self.kernel_size, activation=tf.nn.tanh,padding='same')
 
		pipe=self.batchnorm(pipe,training=True)


		return pipe

#================= Small SMCNN_Unet ========================================#
# Remote: python main.py -mm="ram" --debug=1 -po 1 -ts 1 -tnl 1000000 --batch_size=2000 --filters=256 -pl=5 -m="smcnn_unet" -nap=160000
class SMCNN_UNet(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=256
	def model_graph_get(self,data):

		pipe = data
		pipe = tf.transpose(pipe, [0, 2, 3, 4, 1])
		pipe = tf.reshape(pipe,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(pipe.get_shape())

		pipe = self.conv_block_get(pipe)
		#pipe = self.conv_block_get(pipe)
		#pipe = self.conv_block_get(pipe)				

		pipe = tf.layers.conv2d(pipe, self.n_classes, self.kernel_size, activation=None,padding='same')
		prediction = tf.argmax(pipe, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), pipe
		return pipe, prediction
	def conv_block_get(self,pipe):
		pipe = tf.layers.conv2d(pipe, self.filters, self.kernel_size, activation=None,padding='same')
		pipe=self.batchnorm(pipe,training=self.training)
		pipe = tf.nn.relu(pipe)
		
		return pipe

class SMCNN_UNet_large(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=10
	def model_graph_get(self,data):

		pipe = data
		pipe = tf.transpose(pipe, [0, 2, 3, 4, 1])
		pipe = tf.reshape(pipe,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(pipe.get_shape())

		self.filter_first=64
		#conv0=tf.layers.conv2d(pipe, self.filter_first, self.kernel_size, activation=tf.nn.relu,padding='same')
		conv1=self.conv_block_get(pipe,self.filter_first*2)
		#conv1=self.conv_block_get(pipe,256)
		
		conv2=self.conv_block_get(conv1,self.filter_first*4)
		conv3=self.conv_block_get(conv2,self.filter_first*8)
		##kernel,bias=conv3.variables
		##tf.summary.histogram('conv3', kernel)
		#conv4=self.conv_block_get(conv3,self.filter_first*16)
		#up3=self.deconv_block_get(conv4,conv2,self.filter_first*8)
		
		#self.hidden_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
		#pipe = tf.nn.dropout(pipe, self.keep_prob)
		up4=self.deconv_block_get(conv3,conv2,self.filter_first*4)
		up5=self.deconv_block_get(up4,conv1,self.filter_first*2)
		up6=self.deconv_block_get(up5,pipe,self.filter_first)
		#kernel,bias=up6.variables
		#tf.summary.histogram('up6', kernel)
		
		pipe=self.out_block_get(up6,self.n_classes)
		#pipe = tf.layers.conv2d(up6, self.n_classes, self.kernel_size, activation=None,padding='same')
		prediction = tf.argmax(pipe, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), pipe
		return pipe, prediction

	def conv_2d(self,pipe,filters):
		pipe = tf.layers.conv2d(pipe, filters, self.kernel_size, strides=(1,1), activation=None,padding='same')
		pipe=self.batchnorm(pipe,training=self.training,axis=3)
		
		#pipe=self.batchnorm(pipe,training=True)
		pipe = tf.nn.relu(pipe)
		return pipe
	def conv_block_get(self,pipe,filters):
		
		pipe = self.conv_2d(pipe,filters)
		pipe = self.conv_2d(pipe,filters)
		
		##pipe = tf.layers.conv2d(pipe, filters, self.kernel_size, strides=(1,1), activation=tf.nn.relu,padding='same')
		##pipe = tf.layers.conv2d(pipe, filters, self.kernel_size, strides=(1,1), activation=tf.nn.relu,padding='same')
		
		#pipe = tf.layers.conv2d(pipe, filters, self.kernel_size, strides=(2,2), activation=tf.nn.relu,padding='same')
		pipe=tf.layers.max_pooling2d(inputs=pipe, pool_size=[2, 2], strides=2)
		deb.prints(pipe.get_shape())

		return pipe
	def deconv_2d(self,pipe,filters):
		pipe = tf.layers.conv2d_transpose(pipe, filters, self.kernel_size,strides=(2,2),activation=None,padding='same')
		pipe=self.batchnorm(pipe,training=self.training)
		pipe = tf.nn.relu(pipe)
		return pipe
	def deconv_block_get(self,pipe,layer,filters):
		##pipe = tf.layers.conv2d_transpose(pipe, filters, self.kernel_size,strides=(2,2),activation=tf.nn.relu,padding='same')
		pipe = self.deconv_2d(pipe,filters)
		pipe = tf.concat([pipe,layer],axis=3)
		pipe = self.conv_2d(pipe,filters)
		pipe = self.conv_2d(pipe,filters)
		
		deb.prints(pipe.get_shape())
		return pipe
	def out_block_get(self,pipe,filters):
		pipe = self.conv_2d(pipe,self.filter_first)
		pipe = tf.layers.conv2d(pipe, filters, (1,1), activation=None,padding='same')
		deb.prints(pipe.get_shape())
		return pipe

class SMCNN_UNet(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.kernel_size=(3,3)
		self.gr=16
		self.model_build()
	def conv_block(self,pipe,gr):
		pipe=self.batchnorm(pipe,training=self.training)
		pipe = tf.nn.relu(pipe)
		pipe = tf.layers.conv2d(pipe, gr, self.kernel_size, strides=(1,1), activation=None,padding='same')
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		return pipe

	def dense_block(self,pipe,gr,long_concat=True):
		
		pipe1 = self.conv_block(pipe,gr)
		pipe2 = tf.concat([pipe1,pipe],axis=3)
		pipe2 = self.conv_block(pipe2,gr)
		if long_concat:
			pipe3 = tf.concat([pipe,pipe1,pipe2],axis=3)
		else:
			pipe3 = tf.concat([pipe1,pipe2],axis=3)
	#def transition_block(self,pipe,gr):
		return pipe3

	def model_graph_get(self,data):

		x = data
		x = tf.transpose(x, [0, 2, 3, 4, 1])
		x = tf.reshape(x,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(x.get_shape())

		x = tf.layers.conv2d(x, 48, self.kernel_size, strides=(1,1), activation=None,padding='same')
		deb.prints(x.get_shape())

		pipe2 = self.dense_block(x,self.gr)
		deb.prints(pipe2.get_shape())

		x = self.conv_block(pipe2,80)
		x=tf.layers.average_pooling2d(inputs=x, pool_size=[2, 2], strides=2,padding='same')
		pipe4 = self.dense_block(x,self.gr)
		deb.prints(pipe4.get_shape())

		x = self.conv_block(pipe4,112)
		x=tf.layers.average_pooling2d(inputs=x, pool_size=[2, 2], strides=2,padding='same')
		deb.prints(x.get_shape())

		x = self.dense_block(x,self.gr,long_concat=False)
		deb.prints(x.get_shape())

		x = tf.layers.conv2d_transpose(x, 32, self.kernel_size,strides=(2,2),activation=None,padding='same')
		deb.prints(x.get_shape())

		x = tf.concat([x,pipe4],axis=3)
		
		x = self.dense_block(x,self.gr,long_concat=False)
		x = tf.layers.conv2d_transpose(x, 32, self.kernel_size,strides=(2,2),activation=None,padding='same')
		x = tf.concat([x,pipe2],axis=3)
		x = tf.layers.conv2d(x, self.n_classes, (1,1), activation=None,padding='same')
		prediction = tf.argmax(x, dimension=3, name="prediction")
		return x,prediction


class SMCNN_UNet_small(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.kernel_size=(3,3)
		self.filters=10
	def model_graph_get(self,data):

		pipe = data
		pipe = tf.transpose(pipe, [0, 2, 3, 4, 1])
		pipe = tf.reshape(pipe,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(pipe.get_shape())

		conv1 = tf.layers.conv2d(pipe, 256, self.kernel_size, activation=tf.nn.relu,padding='same')
		conv2 = tf.layers.conv2d(conv1, 256, self.kernel_size, activation=tf.nn.relu,padding='same')
		conv3 = tf.layers.conv2d(conv2, 256, self.kernel_size, activation=tf.nn.relu,padding='same')
		#conv4 = tf.layers.conv2d(conv3, 256, self.kernel_size, activation=tf.nn.relu,padding='same')

		layer_in = tf.concat([conv4,conv2],axis=3)
		conv5 = tf.layers.conv2d(layer_in, 256, self.kernel_size, activation=tf.nn.relu,padding='same')

		layer_in = tf.concat([conv5,conv1],axis=3)
		pipe = tf.layers.conv2d(layer_in, self.n_classes, self.kernel_size, activation=None,padding='same')

		prediction = tf.argmax(pipe, dimension=3, name="prediction")
		#return tf.expand_dims(annotation_pred, dim=3), pipe
		return pipe, prediction
	def conv_block_get(self,pipe):
		pipe = tf.layers.conv2d(pipe, self.filters, self.kernel_size, activation=None,padding='same')
		pipe=self.batchnorm(pipe,training=self.training)
		pipe = tf.nn.relu(pipe)
		
		return pipe


# ================================= Implements ConvLSTM ============================================== #
class conv_lstm_semantic(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data): #self.kernel
		
		pipe1=self.layer_lstm_get(data,filters=20,kernel=[3,3],name='convlstm')
		#pipe1=self.layer_lstm_multi_get(data,filters=20,kernel=[3,3],name='convlstm')
		
		tf.summary.histogram("lstm_out1",pipe1)
		tf.summary.image("lstm_out",tf.cast(tf.squeeze(tf.gather(pipe1,[0,1,2],axis=3)),tf.uint8))

		if self.debug: deb.prints(pipe1.get_shape())

		self.layer_idx=0
		pipe=self.conv2d_block_get(pipe1,20,training=self.training,layer_idx=self.layer_idx,kernel=2)
		#self.layer_idx+=1
		
		#pipe=self.resnet_block_get(pipe1,20,training=self.training,layer_idx=self.layer_idx,kernel=2)
		self.layer_idx+=1

		if self.debug: deb.prints(pipe.get_shape())
		pipe = tf.concat([pipe1,pipe],axis=3)

		pipe,prediction=self.conv2d_out_get(pipe,self.n_classes,kernel_size=1,layer_idx=self.layer_idx)
		self.layer_idx+=1
		if self.debug: deb.prints(pipe.get_shape())
		return pipe,prediction

class conv_lstm_semantic(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data): #self.kernel
		
		pipe1=self.layer_lstm_get(data,filters=20,kernel=[3,3],name='convlstm')
		#pipe1=self.layer_lstm_multi_get(data,filters=20,kernel=[3,3],name='convlstm')
		
		tf.summary.histogram("lstm_out1",pipe1)
		tf.summary.image("lstm_out",tf.cast(tf.squeeze(tf.gather(pipe1,[0,1,2],axis=3)),tf.uint8))

		self.layer_idx=0
		pipe={'down':[], 'up':[]}
		c={'down':0, 'up':0}
		filters=20
		pipe['down'].append(self.transition_down(pipe1,filters)) #0 16x16
		pipe['down'].append(self.transition_down(pipe['down'][0],filters)) #1 8x8
		pipe['down'].append(self.transition_down(pipe['down'][1],filters)) #2 4x4
		
		pipe['down'].append(self.dense_block(pipe['down'][2],filters)) #3 4x4

		pipe['up'].append(self.concatenate_transition_up(pipe['down'][3],pipe['down'][2],filters)) # 0 8x8
		pipe['up'].append(self.concatenate_transition_up(pipe['up'][0],pipe['down'][1],filters)) # 1
		pipe['up'].append(self.concatenate_transition_up(pipe['up'][1],pipe['down'][0],filters)) # 2
		self.layer_idx+=1
		out,prediction=self.conv2d_out_get(pipe['up'][-1],self.n_classes,kernel_size=1,layer_idx=self.layer_idx)
		self.layer_idx+=1
		if self.debug: deb.prints(out.get_shape())
		return out,prediction
# ================================= Implements SMCNN ============================================== #
# Remote: python main.py -mm="ram" --debug=1 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256 -m="smcnn"
class SMCNN_semantic(NeuralNetSemantic):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		# Data if of shape [None,6,32,32,6]
		pipe = data
		pipe = tf.transpose(pipe, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		pipe = tf.reshape(pipe,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]

		deb.prints(pipe.get_shape())

		pipe = tf.layers.conv2d(pipe, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		if self.debug: deb.prints(pipe.get_shape())
		pipe = tf.layers.conv2d(pipe, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		self.layer_idx=10
		pipe,prediction=self.conv2d_out_get(pipe,self.n_classes,kernel_size=1,layer_idx=self.layer_idx)
		self.layer_idx+=1
		if self.debug: deb.prints(pipe.get_shape())
		return pipe,prediction




# ================================= Implements Conv3DMultitemp ============================================== #
class Conv3DMultitemp(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		print(1)
		super().__init__(*args, **kwargs)
		self.kernel=[3,3,3]
		deb.prints(self.kernel)
		self.model_build()
		
	def model_graph_get(self,data):
		pipe=self.layer_lstm_get(data,filters=self.filters,kernel=[3,3],get_last=False,name="convlstm")
		#pipe=tf.layers.conv3d(pipe,self.filters,[1,3,3],padding='same',activation=tf.nn.tanh)
		if self.debug: deb.prints(pipe.get_shape())
		pipe=tf.layers.conv3d(pipe,16,[3,3,3],padding='same',activation=tf.nn.tanh)
		
		pipe=tf.layers.max_pooling3d(inputs=pipe, pool_size=[2,1,1], strides=[2,1,1],padding='same')
		if self.debug: deb.prints(pipe.get_shape())

		#pipe=tf.layers.conv3d(pipe,self.filters,[],padding='same',activation=tf.nn.tanh)
		
		#pipe=tf.layers.conv3d(data,self.filters,self.kernel,padding='same',activation=tf.nn.tanh)
		
		#pipe=tf.layers.max_pooling2d(inputs=pipe, pool_size=[2, 2], strides=2)
		#pipe = tf.layers.conv2d(pipe, self.filters, self.kernel_size, activation=tf.nn.tanh)
		pipe = tf.contrib.layers.flatten(pipe)
		if self.debug: deb.prints(pipe.get_shape())
		pipe = tf.layers.dense(pipe, 128,activation=tf.nn.tanh)
		if self.debug: deb.prints(pipe.get_shape())
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())
		return None,pipe
# ================================= Implements SMCNN ============================================== #
# Remote: python main.py -mm="ram" --debug=1 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256 -m="smcnn"
class SMCNN(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		# Data if of shape [None,6,32,32,6]
		pipe = data
		pipe = tf.transpose(pipe, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		pipe = tf.reshape(pipe,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]

		deb.prints(pipe.get_shape())

		pipe = tf.layers.conv2d(pipe, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe=tf.layers.max_pooling2d(inputs=pipe, pool_size=[2, 2], strides=2)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.contrib.layers.flatten(pipe)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.layers.dense(pipe, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(pipe.get_shape())
		
		#pipe = tf.layers.dropout(pipe,rate=self.keep_prob,training=False,name='dropout')
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())


		return None,pipe

# ================================= Implements SMCNN ============================================== #
class SMCNNlstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		
	def model_graph_get(self,data):
		#pipe = tf.gather(data, int(data.get_shape()[1]) - 1,axis=1)
		pipe = data

		pipe=self.layer_lstm_get(data,filters=self.filters,kernel=[3,3],get_last=False,name="convlstm")

		pipe = tf.transpose(data, [0, 2, 3, 4, 1])
		pipe = tf.reshape(pipe,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps])

		deb.prints(pipe.get_shape())

		#pipe = tf.layers.conv2d(pipe, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		#if self.debug: deb.prints(pipe.get_shape())
		
		pipe=tf.layers.max_pooling2d(inputs=pipe, pool_size=[2, 2], strides=2)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.contrib.layers.flatten(pipe)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.layers.dense(pipe, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(pipe.get_shape())
		
		#pipe = tf.layers.dropout(pipe,rate=self.keep_prob,training=False,name='dropout')
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())


		return None,pipe

class SMCNNlstm(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model_build()
		self.activations=tf.nn.relu
	def model_graph_get(self,data):
		# Data if of shape [None,6,32,32,6]
		pipe1 = data
		pipe1 = tf.transpose(pipe1, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		pipe1 = tf.reshape(pipe1,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]
		deb.prints(pipe1.get_shape())
		#pipe1 = tf.layers.conv2d(pipe1, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		if self.debug: deb.prints(pipe1.get_shape())
		
		pipe2=self.layer_lstm_get(data,filters=5,kernel=[3,3],get_last=True,name="convlstm")
		deb.prints(pipe2.get_shape())
		#pipe1 = tf.transpose(pipe1, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		#pipe1 = tf.reshape(pipe1,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]
		

		pipe=tf.concat([pipe1,pipe2],axis=3)

		deb.prints(pipe.get_shape())
		pipe = tf.layers.conv2d(pipe, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		
		
		pipe=tf.layers.max_pooling2d(inputs=pipe, pool_size=[2, 2], strides=2)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.contrib.layers.flatten(pipe)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.layers.dense(pipe, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(pipe.get_shape())
		
		#pipe = tf.layers.dropout(pipe,rate=self.keep_prob,training=False,name='dropout')
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())


		return None,pipe
# ================================= Implements SMCNN ============================================== #
# Remote: python main.py -mm="ram" --debug=1 -po 4 -ts 1 -tnl 10000000 -bs=20000 --batch_size=2000 --filters=256 -m="smcnn"
class SMCNN_conv3d(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.kernel=[3,3,3]
		self.model_build()
		
	def model_graph_get(self,data):
		# Data if of shape [None,6,32,32,6]
		pipe = data
		#pipe = tf.transpose(pipe, [0, 2, 3, 4, 1]) # Transpose shape [None,32,32,6,6]
		#pipe = tf.reshape(pipe,[-1,self.patch_len,self.patch_len,self.channels*self.timesteps]) # Shape [None,32,32,6*6]

		deb.prints(pipe.get_shape())

		#pipe = tf.layers.conv2d(pipe, 256, self.kernel_size, activation=tf.nn.tanh,padding="same")
		pipe=tf.layers.conv3d(pipe,256,[3,3,3],padding='same',activation=tf.nn.tanh)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe=tf.layers.max_pooling3d(inputs=pipe, pool_size=[2,1,1], strides=[2,1,1],padding='same')
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.contrib.layers.flatten(pipe)
		if self.debug: deb.prints(pipe.get_shape())
		
		pipe = tf.layers.dense(pipe, 256,activation=tf.nn.tanh,name='hidden')
		if self.debug: deb.prints(pipe.get_shape())
		
		#pipe = tf.layers.dropout(pipe,rate=self.keep_prob,training=False,name='dropout')
		pipe = tf.nn.dropout(pipe, self.keep_prob)
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())


		return None,pipe




class Conv3DMultitemp(NeuralNetOneHot):
	def __init__(self, *args, **kwargs):
		print(1)
		super().__init__(*args, **kwargs)
		self.kernel=[3,3,3]
		deb.prints(self.kernel)
		self.model_build()
		
	def model_graph_get(self,data):
		pipe=self.layer_lstm_get(data,filters=self.filters,kernel=[3,3],get_last=False,name="convlstm")
		#pipe=tf.layers.conv3d(pipe,self.filters,[1,3,3],padding='same',activation=tf.nn.tanh)
		if self.debug: deb.prints(pipe.get_shape())
		pipe=tf.layers.conv3d(pipe,16,[3,3,3],padding='same',activation=tf.nn.tanh)
		
		pipe=tf.layers.max_pooling3d(inputs=pipe, pool_size=[2,1,1], strides=[2,1,1],padding='same')
		if self.debug: deb.prints(pipe.get_shape())

		#pipe=tf.layers.conv3d(pipe,self.filters,[],padding='same',activation=tf.nn.tanh)
		
		#pipe=tf.layers.conv3d(data,self.filters,self.kernel,padding='same',activation=tf.nn.tanh)
		
		#pipe=tf.layers.max_pooling2d(inputs=pipe, pool_size=[2, 2], strides=2)
		#pipe = tf.layers.conv2d(pipe, self.filters, self.kernel_size, activation=tf.nn.tanh)
		pipe = tf.contrib.layers.flatten(pipe)
		if self.debug: deb.prints(pipe.get_shape())
		pipe = tf.layers.dense(pipe, 128,activation=tf.nn.tanh)
		if self.debug: deb.prints(pipe.get_shape())
		pipe = tf.layers.dense(pipe, self.n_classes,activation=tf.nn.softmax)
		if self.debug: deb.prints(pipe.get_shape())
		return None,pipe



