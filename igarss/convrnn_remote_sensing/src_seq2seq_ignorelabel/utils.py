#import skimage.io as io
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def read_img(img_path):
	img = io.imread(img_path)
	return img


def save_img(img_path, np_array):
	io.imsave(img_path, np_array)


def extract_patches(output_folder, img_array, img_reference, stride):
	# Hint: create a rutine to extract patches from training image and training reference"
	# Create a tensor [rows, cols, R_G_B_Labels]
	# save patches to train folder
	np.save(output_folder + patch_name, patch_array_tensor)


def normalize(im_patch_array):
    return np.array(im_patch_array)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def load_data(RGBLabel_patches_path_npy):
    RGBLabels = np.load(RGBLabel_patches_path_npy)
    rgb_patch = RGBLabels[:, :, :3]
    rgb_patch = normalize(rgb_patch)
    label_patch = RGBLabels[:, :, 3]
    return np.concatenate((rgb_patch, label_patch), axis=2)


def compute_metrics(true_labels, predicted_labels):
	accuracy = accuracy_score(true_labels, predicted_labels)
	f1score = 100*f1_score(true_labels, predicted_labels, average=None)
	recall = 100*recall_score(true_labels, predicted_labels, average=None)
	prescision = 100*precision_score(true_labels, predicted_labels, average=None)
	return accuracy, f1score, recall, prescision
