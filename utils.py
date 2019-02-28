import os
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

from read_nc import get_data
import configparser


def get_names(config):
	return os.listdir(config.input)


def is_zero(l):
	for i in l:
		if max(i) > 10:
			return True
	return False


def preprocess_reconstruct(config):
	factors_data = get_data(config.input)
	dem_data = get_data(config.feature)

	# 将nan数据置零
	dex = np.isnan(dem_data)
	dem_data[dex] = 0
	dex = np.isnan(factors_data)
	factors_data[dex] = 0
	# dem_data = dem_data/3000.

	# img_ = get_data(config.input)
	scale0w = config.factors_datasize_w / factors_data.shape[-2]
	scale0j = config.factors_datasize_j / factors_data.shape[-1]
	img_ = scipy.ndimage.zoom(factors_data, (1, scale0w, scale0j))

	scale1w = config.factors_datasize_w / dem_data.shape[-2]
	scale1j = config.factors_datasize_j / dem_data.shape[-1]
	dem_data = scipy.ndimage.zoom(dem_data, (1, scale1w, scale1j))

	shap = img_.shape
	image_ = np.zeros((shap[0], shap[1], shap[2], 2))
	image_[:, :, :, 0] = img_
	image_[:, :, :, 1] = dem_data
	return image_


def preprocess(config):
	dem_data = get_data(config.dem_dir)                             # shape(1, size_w, size_j)
	label_data = get_data(config.factors_dir, config.factor_str)    # shape(72_time, size_w, size_j)
	factors_data = get_data(config.factors_dir)                     # shape(6, 72_time, size_w, size_j)


	print(factors_data.shape, dem_data.shape, label_data.shape)

	# 将nan数据置零
	# dex = np.isnan(dem_data)
	# dem_data[dex] = 0
	# dex = np.isnan(factors_data)
	# factors_data[dex] = 0
	# # dem_data = dem_data / 3000.
	#
	# # filter the data
	# print("before filter input size is:")
	# print(factors_data.shape)
	# factors_data = np.array(list(filter(is_zero, factors_data)))
	# print("after filter the input size is:")
	# print(factors_data.shape)
	#
	# label_ = factors_data
	# print("after filter the label size is:")
	# print(label_.shape)
	#
	# scale0w = config.factors_datasize_w / factors_data.shape[-2]
	# scale0j = config.factors_datasize_j / factors_data.shape[-1]
	# img_ = scipy.ndimage.zoom(factors_data, (1, scale0w, scale0j))
	#
	# img_ = np.maximum(img_, 0)
	# img_ = np.where(img_ < 0, 0, img_)
	# img_ = np.where(img_ > 9999, 0, img_)
	#
	# scale1w = config.factors_datasize_w / dem_data.shape[-2]
	# scale1j = config.factors_datasize_j / dem_data.shape[-1]
	# dem_data = scipy.ndimage.zoom(dem_data, (1, scale1w, scale1j))
	#
	# scale2w = config.output_size_w / label_.shape[-2]
	# scale2j = config.output_size_j / label_.shape[-1]
	# label_ = scipy.ndimage.zoom(label_, (1, scale2w, scale2j))
	# label_ = np.maximum(label_, 0)
	# label_ = label_.reshape(-1, config.output_size_w, config.output_size_j, 1)
	#
	# shap = img_.shape
	# image_ = np.zeros((shap[0], shap[1], shap[2], 2))
	# image_[:, :, :, 0] = img_
	# image_[:, :, :, 1] = dem_data
	# return image_, label_


def parse_config_test(config_path, section, option):
	cp = configparser.ConfigParser()

	if not os.path.exists(config_path):
		print("NO configuration:{0} !".format(config_path))
		exit()
	cp.read(config_path)
	return cp.get(section, option)

# if __name__=='__main__':
# parse_config()


