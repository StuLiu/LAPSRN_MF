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
	factors_data = get_data(config.factors_dir)
	dem_data = get_data(config.dem_dir)

	# 将nan数据置零
	dex = np.isnan(dem_data)
	dem_data[dex] = 0
	dex = np.isnan(factors_data)
	factors_data[dex] = 0
	# dem_data = dem_data/3000.

	# img_ = get_data(config.input)
	scale_w = config.input_size_w / factors_data.shape[-2]
	scale_j = config.input_size_j / factors_data.shape[-1]
	factors_data_scaled = scipy.ndimage.zoom(factors_data, (1, scale_w, scale_j))
	print('factors_data_scaled:', factors_data_scaled)

	dem_data_scaled = scipy.ndimage.zoom(dem_data, (1, scale_w, scale_j))
	dem_data_scaled = np.tile(dem_data_scaled, (factors_data_scaled.shape[1], 1))  # 重复时间维次:factors_data_scaled.shape[1]
	dem_data_scaled = dem_data_scaled.reshape(1, factors_data_scaled.shape[1], config.input_size_w, config.input_size_j)
	print('dem_data_scaled:', dem_data_scaled)

	input_ = np.append(factors_data_scaled, dem_data_scaled)
	input_ = input_.reshape(7, factors_data_scaled.shape[1], config.input_size_w, config.input_size_j)
	input_ = input_.transpose((1, 2, 3, 0))
	return input_


def preprocess(config):
	dem_data = get_data(config.dem_dir)                             # shape = (1, size_w, size_j)
	label_data = get_data(config.factors_dir, config.factor_str)    # shape = (72_time, size_w, size_j)
	factors_data = get_data(config.factors_dir)                     # shape = (6, 72_time, size_w, size_j)
	print(dem_data.shape, label_data.shape, factors_data.shape)

	# 将nan数据置零
	dex = np.isnan(dem_data)
	dem_data[dex] = 0
	dex = np.isnan(factors_data)
	factors_data[dex] = 0
	# dem_data = dem_data / 3000.

	# # filter the data
	# print("before filter input size is:")
	# print("factors_data.shape:", factors_data.shape)
	# factors_data = np.array(list(filter(is_zero, factors_data)))
	# print("after filter the input size is:")
	# print(factors_data.shape)

	# label_ = factors_data
	# print("after filter the label size is:")
	# print(label_.shape)

	# 缩放各要素数据，作为模型输入的一部分
	scale_w = config.input_size_w / factors_data.shape[-2]
	scale_j = config.input_size_j / factors_data.shape[-1]
	factors_data_scaled = scipy.ndimage.zoom(factors_data, (1, 1, scale_w, scale_j))
	factors_data_scaled = np.maximum(factors_data_scaled, 0)
	factors_data_scaled = np.where(factors_data_scaled < 0, 0, factors_data_scaled)
	factors_data_scaled = np.where(factors_data_scaled > 9999, 0, factors_data_scaled)
	# factors_data_scaled shape = (6, 72_time, input_size_w, input_size_j)
	print('factors_data_scaled.shape:', factors_data_scaled.shape)

	# 缩放地形数据，作为模型输入的一部分
	dem_data = scipy.ndimage.zoom(dem_data, (1, scale_w, scale_j))
	dem_data = np.tile(dem_data, (factors_data_scaled.shape[1], 1))  # 重复时间维次:factors_data_scaled.shape[1]
	dem_data = dem_data.reshape(1, factors_data_scaled.shape[1], config.input_size_w, config.input_size_j)
	# dem_data shape = (1, 72_time, input_size_w, input_size_j)
	print('dem_data.shape:', dem_data.shape)

	# 缩放标签数据，作为模型输入的label部分
	scale_w = config.output_size_w / label_data.shape[-2]
	scale_j = config.output_size_j / label_data.shape[-1]
	label_ = scipy.ndimage.zoom(label_data, (1, scale_w, scale_j))
	label_ = np.maximum(label_, 0)
	label_ = label_.reshape(-1, config.output_size_w, config.output_size_j, 1)
	# label_ shape = (72_time, output_size_w, output_size_j)
	print('label_.shape:', label_.shape)

	input_ = np.append(factors_data_scaled, dem_data)
	input_ = input_.reshape(7, factors_data_scaled.shape[1], config.input_size_w, config.input_size_j)
	print('input_.shape origin:', input_.shape)
	input_ = input_.transpose((1, 2, 3, 0))
	print('input_.shape transposed:', input_.shape)
	return input_, label_


def parse_config_test(config_path, section, option):
	cp = configparser.ConfigParser()

	if not os.path.exists(config_path):
		print("NO configuration:{0} !".format(config_path))
		exit()
	cp.read(config_path)
	return cp.get(section, option)

if __name__=='__main__':
	a = np.array([[[1, 2, 3], [4, 5, 6]]])
	b = np.array([
					[[7, 8, 9],
	                    [10, 11, 12]],
					[[7, 8, 9],
					    [10, 11, 12]]
	              ])
	print(np.append(a,b).reshape(3, 2 ,3).transpose(1,2,0))