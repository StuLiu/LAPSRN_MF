import os
import scipy.misc
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt

from read_nc import *
import configparser

def preprocess_reconstruct(config):
	PRE_data = read_factor(config.input_dir, 'PRE10m') / 30
	RHU_data = read_factor(config.input_dir, 'RHU') / 100
	# label_data = None
	if config.factor_str == 'PRE10m':
		PRE_data *= 30
		label_data = PRE_data  # shape = (time, size_w, size_j)
	elif config.factor_str == 'RHU':
		RHU_data *= 100
		label_data = RHU_data
	else:
		raise Exception('No such weather factor')
	print('label_data.shape:', label_data.shape)

	dem_data = read_dem(config.dem_path) / 3000  # shape = (1, size_w, size_j)
	dem_data = np.tile(dem_data, (label_data.shape[0], 1))  # shape = (time, size_w, size_j)
	dem_data = dem_data.reshape(-1, label_data.shape[-2], label_data.shape[-1])
	print('dem_data.shape:', dem_data.shape)

	input_data = np.array([PRE_data, RHU_data, dem_data])  # shape = (3, time, size_w, size_j)
	input_data = input_data.reshape(-1, label_data.shape[0], label_data.shape[-2], label_data.shape[-1])
	print('input_data.shape:', input_data.shape)
	plt.imsave('example/test_origin_PRE.png', input_data[0, 0, :, :])
	plt.imsave('example/test_origin_RHU.png', input_data[1, 0, :, :])
	plt.imsave('example/test_origin_dem.png', input_data[2, 0, :, :])

	# 缩放input数据，作为模型的输入X
	scale_w = config.input_size_w / input_data.shape[-2]
	scale_j = config.input_size_j / input_data.shape[-1]
	input_data = scipy.ndimage.zoom(input_data, (1, 1, scale_w, scale_j))
	input_data = input_data.reshape(-1, label_data.shape[0], config.input_size_w, config.input_size_j)
	input_data = input_data.transpose((1, 2, 3, 0))
	print('scaled input_data.shape:', input_data.shape)  # shape = (time, size_w, size_j, 2)
	plt.imsave('example/test_input_PRE.png', input_data[0, :, :, 0])
	plt.imsave('example/test_input_RHU.png', input_data[0, :, :, -2])
	plt.imsave('example/test_input_dem.png', input_data[0, :, :, -1])
	return input_data

def preprocess(config):
	PRE_data = read_factor(config.factors_dir, 'PRE10m') / 30
	RHU_data = read_factor(config.factors_dir, 'RHU') / 100
	# label_data = None
	if config.factor_str == 'PRE10m':
		PRE_data *= 30
		label_data = PRE_data                               # shape = (time, size_w, size_j)
	elif config.factor_str == 'RHU':
		RHU_data *= 100
		label_data = RHU_data
	else:
		raise Exception('No such weather factor')
	print('label_data.shape:', label_data.shape)
	dem_data = read_dem(config.dem_path) / 3000             # shape = (1, size_w, size_j)
	dem_data = np.tile(dem_data, (label_data.shape[0], 1))  # shape = (time, size_w, size_j)
	dem_data = dem_data.reshape(-1, label_data.shape[-2], label_data.shape[-1])
	print('dem_data.shape:', dem_data.shape)
	input_data = np.array([PRE_data, RHU_data, dem_data])   # shape = (3, time, size_w, size_j)
	input_data = input_data.reshape(-1, label_data.shape[0], label_data.shape[-2], label_data.shape[-1])
	print('input_data.shape:', input_data.shape)
	plt.imsave('example/train_origin_PRE.png', input_data[0, 0, :, :])
	plt.imsave('example/train_origin_RHU.png', input_data[1, 0, :, :])
	plt.imsave('example/train_origin_dem.png', input_data[2, 0, :, :])
	# 缩放input数据，作为模型的输入X
	scale_w = config.input_size_w / input_data.shape[-2]
	scale_j = config.input_size_j / input_data.shape[-1]
	input_data = scipy.ndimage.zoom(input_data, (1, 1, scale_w, scale_j))
	input_data = input_data.reshape(-1, label_data.shape[0], config.input_size_w, config.input_size_j)
	input_data = input_data.transpose((1, 2, 3, 0))
	print('scaled input_data.shape:', input_data.shape)     # shape = (time, size_w, size_j, 2)
	plt.imsave('example/train_input_PRE.png', input_data[0, :, :, 0])
	plt.imsave('example/train_input_RHU.png', input_data[0, :, :, -2])
	plt.imsave('example/train_input_dem.png', input_data[0, :, :, -1])
	# 缩放标签数据，作为模型的输入Y
	scale_w = config.output_size_w / label_data.shape[-2]
	scale_j = config.output_size_j / label_data.shape[-1]
	label_data = scipy.ndimage.zoom(label_data, (1, scale_w, scale_j))
	label_data = label_data.reshape(-1, config.output_size_w, config.output_size_j, 1)
	print('scaled label.shape:', label_data.shape)          # shape = (time, size_w, size_j, 1)
	plt.imsave('example/train_label_RHU.png', label_data[0, :, :, 0])
	return input_data, label_data


def parse_config_test(config_path, section, option):
	cp = configparser.ConfigParser()

	if not os.path.exists(config_path):
		print("NO configuration:{0} !".format(config_path))
		exit()
	cp.read(config_path)
	return cp.get(section, option)

# def __normalize(input):
# 	if len(input.shape) != 4 or input.shape[-1] != 3:
# 		raise Exception('normalize argument shape error')
# 	# shape = (7, 72_time, size_w, size_j)
# 	input[0] = input[0] / 30   # 'PRE10m'
# 	# input[1] = input[1] / 1000  # 'PRS'
# 	input[1] = input[1] / 100   # 'RHU'
# 	# input[3] = input[3] / 40    # 'TEM'
# 	# input[4] = input[4] / 360   # 'WINDAvg2mi'
# 	# input[5] = input[5] / 10    # 'WINSAvg2mi'
# 	# input[6] = input[6] / 3000  # 'dem'
# 	return input

def get_names(config):
	return os.listdir(config.input)

def is_zero(l):
	for i in l:
		if max(i) > 10:
			return True
	return False


if __name__=='__main__':
	a = np.array([[[1, 2, 3], [4, 5, 6]]])
	b = np.array([
					[[7, 8, 9],
	                    [10, 11, 12]],
					[[7, 8, 9],
					    [10, 11, 12]]
	              ])
	print(np.append(a,b).reshape(3, 2 ,3).transpose(1,2,0))