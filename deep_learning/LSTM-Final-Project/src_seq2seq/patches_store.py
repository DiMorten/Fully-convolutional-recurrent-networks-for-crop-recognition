

from __future__ import division
import os
import math
import random
import numpy as np
from time import gmtime, strftime
import glob
import tensorflow as tf
import numpy as np
from random import shuffle
import glob
import sys
import pickle
import argparse

# Local
import utils
import deb
from model import (conv_lstm,Conv3DMultitemp,UNet,SMCNN,SMCNNlstm, SMCNN_UNet, SMCNN_conv3d, lstm, conv_lstm_semantic, SMCNN_semantic)

#Input configuration
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train', help='phase')
parser.add_argument('--dataset_name', dest='dataset_name', default='20160419', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=200, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--timesteps', dest='timesteps', type=int, default=7, help='# timesteps used to train')
parser.add_argument('-pl','--patch_len', dest='patch_len', type=int, default=5, help='# timesteps used to train')
parser.add_argument('--kernel', dest='kernel', type=int, default=[3,3], help='# timesteps used to train')
#parser.add_argument('--channels', dest='channels', type=int, default=7, help='# timesteps used to train')
parser.add_argument('--filters', dest='filters', type=int, default=32, help='# timesteps used to train')
#parser.add_argument('--n_classes', dest='n_classes', type=int, default=6, help='# timesteps used to train')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('-m','--model', dest='model', default='convlstm', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='../data/summaries/', help='models are saved here')

parser.add_argument('--debug', type=int, default=1, help='Debug')
parser.add_argument('-po','--patch_overlap', dest='patch_overlap', type=int, default=0, help='Debug')
parser.add_argument('--im_size', dest='im_size',default=[948,1068], help='Debug')
parser.add_argument('--im_h', type=int, dest='im_h',default=948, help='Debug')
parser.add_argument('--im_w', type=int, dest='im_w',default=1068, help='Debug')

parser.add_argument('--band_n', dest='band_n', type=int, default=7, help='Debug')
parser.add_argument('--t_len', dest='t_len', type=int, default=6, help='Debug')
parser.add_argument('--path', dest='path', default="../data/", help='Data path')
parser.add_argument('--class_n', dest='class_n', type=int, default=6, help='Class number')
parser.add_argument('--pc_mode', dest='pc_mode', default="local", help="Class number. 'local' or 'remote'")
parser.add_argument('-tnl','--test_n_limit', dest='test_n_limit',type=int, default=1000, help="Class number. 'local' or 'remote'")
parser.add_argument('-mm','--memory_mode', dest='memory_mode',default="ram", help="Class number. 'local' or 'remote'")
parser.add_argument('-bs','--balance_samples_per_class', dest='balance_samples_per_class',type=int,default=None, help="Class number. 'local' or 'remote'")
parser.add_argument('-ts','--test_get_stride', dest='test_get_stride',type=int,default=8, help="Class number. 'local' or 'remote'")
parser.add_argument('-nap','--n_apriori', dest='n_apriori',type=int,default=4000000, help="Class number. 'local' or 'remote'")
parser.add_argument('-sc','--squeeze_classes', dest='squeeze_classes',default=True, help="Class number. 'local' or 'remote'")


parser.add_argument('-tof','--test_overlap_full', dest='test_overlap_full',default=False, help="Class number. 'local' or 'remote'")
parser.add_argument('-fes','--fine_early_stop', dest='fine_early_stop',default=True, help="Class number. 'local' or 'remote'")
parser.add_argument('-ttmn','--train_test_mask_name', dest='train_test_mask_name',default="TrainTestMask.tif", help="Class number. 'local' or 'remote'")
parser.add_argument('--id_first', dest='id_first', type=int, default=1, help='Class number')
parser.add_argument('-ir','--im_reconstruct', dest='im_reconstruct',default=False, help="Class number. 'local' or 'remote'")

parser.add_argument('-rst','--ram_store', dest='ram_store',default=True, help="Ram store")
parser.add_argument('-psv','--patches_save', dest='patches_save',default=True, help="Patches npy store")
parser.add_argument('-lf','--label_folder', dest='label_folder',default='labels', help="Label folder")

args = parser.parse_args()

args.n_classes=args.class_n
args.timesteps=args.t_len
args.channels=args.band_n
np.set_printoptions(suppress=True)
os.system("rm -rf ../"+args.path+"/summaries/*")

# Check if selected model has one_hot (One pixel) or semantic (Image) output type
if args.model=='unet' or args.model=='smcnn_unet' or args.model=='convlstm_semantic' or args.model=='smcnn_semantic':
    label_type='semantic'
else:
    label_type='one_hot'
deb.prints(label_type)
deb.prints(args.patches_save)

def main(_):

    # Make checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Create a dataset object
    if label_type=='one_hot':
        data=utils.DataOneHot(debug=args.debug, patch_overlap=args.patch_overlap, im_size=args.im_size, \
                                band_n=args.band_n, t_len=args.t_len, path=args.path, class_n=args.class_n, pc_mode=args.pc_mode, \
                                test_n_limit=args.test_n_limit,memory_mode=args.memory_mode, \
                                balance_samples_per_class=args.balance_samples_per_class, test_get_stride=args.test_get_stride, \
                                n_apriori=args.n_apriori,patch_length=args.patch_len,squeeze_classes=args.squeeze_classes,im_h=args.im_h,im_w=args.im_w, \
                                id_first=args.id_first, train_test_mask_name=args.train_test_mask_name, \
                                test_overlap_full=args.test_overlap_full,ram_store=args.ram_store,patches_save=args.patches_save)
    elif label_type=='semantic':
        data=utils.DataSemantic(debug=args.debug, patch_overlap=args.patch_overlap, im_size=args.im_size, \
                                band_n=args.band_n, t_len=args.t_len, path=args.path, class_n=args.class_n, pc_mode=args.pc_mode, \
                                test_n_limit=args.test_n_limit,memory_mode=args.memory_mode, \
                                balance_samples_per_class=args.balance_samples_per_class, test_get_stride=args.test_get_stride, \
                                n_apriori=args.n_apriori,patch_length=args.patch_len,squeeze_classes=args.squeeze_classes,im_h=args.im_h,im_w=args.im_w, \
                                id_first=args.id_first, train_test_mask_name=args.train_test_mask_name, \
                                test_overlap_full=args.test_overlap_full,ram_store=args.ram_store,patches_save=args.patches_save,label_folder=args.label_folder)


    # Load images and create dataset (Extract patches)
    if args.memory_mode=="ram":
        data.create()
        deb.prints(data.ram_data["train"]["ims"].shape)



if __name__ == '__main__':
    tf.app.run()



