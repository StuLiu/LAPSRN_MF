import netCDF4 as nc
import os
import glob
import numpy as np


'''获取文件夹里，nc文件的路径列表'''
def prepare_data(dataset):
	print(dataset)
	if not os.path.exists(dataset):
		print(" No such file or directory:{0}".format(dataset))
		exit()
	#获取ｎｃ文件路径列表
	filenames = os.listdir(dataset)
	# os.getcwd()：获取当前工作目录，也就是在哪个目录下运行这个程序。
	data_dir = os.path.join(os.getcwd(), dataset)
	# glob函数用来查找符合指定规则的文件路径名
	data = glob.glob(os.path.join(data_dir, "*.nc"))
	data.sort()
	return data

def get_time_dimen(config):
	paths = prepare_data(config.input_dir)
	if len(paths) == 0:
		print("There is no file in the:{0}".format(config.input_dir))
		exit()
	path = paths[0]
	
	dataset = nc.Dataset(path)
	keys = list(dataset.variables.keys())
	keys = np.sort(keys)
	
	time = dataset.variables[keys[-1]][:]
	# *60.0 means change hours into minutes
	time = time * 60.0
	return time

def get_lat_lon_dimen(config):
	paths = prepare_data(config.factors_dir)
	if len(paths) == 0:
		print("There is no file in the:{0}".format(config.factors_dir))
		exit()
	path = paths[0]
	dataset = nc.Dataset(path)
	lat = dataset.variables["lat"][:]
	lon = dataset.variables["lon"][:]
	dataset.close()
	return lat, lon

def get_name(name):
	paths = prepare_data(name)
	paths = sorted(paths)
	if len(paths) == 0:
		print("There is no file in the:{0}".format(name))
		exit()
	dataset = nc.Dataset(paths[0])
	data = dataset.variables["time"][:]
	predit_hour = str(data[-1])
	names = [path.split('/')[-1] for path in paths]
	names_new = []
	for name in names:
		l = name.split("_")
		mode_str = l[0]
		time_str = l[1] + "00"
		area_str = l[-1].split(".")[0]
		new_name = "MSP2_PMSC_AIWSRPF_" + mode_str + "SP1_L88_" + area_str + "_" +  time_str + "_00000-0{0}00.nc".format(predit_hour)
		names_new.append(new_name)
	return names_new

def get_data(name, rang=''):
	print('getData')
	paths = prepare_data(name)
	paths = sorted(paths)
	print(paths)
	d = []
	if (len(rang) == 0):
		for path in paths:
			#print(path,' : ',progress(path).shape)
			d.append(progress(path))
		d = np.array(d)  # .squeeze()
		if d.shape[0] == 1:     # 读取地形
			d = d.reshape(-1, d.shape[-2], d.shape[-1])
		elif d.shape[0] == 6:   # 读取6个要素数据，依次为降水、压强、湿度、温度、风向、风速。
			d = d.reshape(6, -1, d.shape[-2], d.shape[-1]) # (气象要素, 时间维, 纬度, 经度)
	else:
		for path in paths:
			if (rang in path):
				d.append(progress(path))
				d = np.array(d)#.squeeze()
				d = d.reshape(-1, d.shape[-2], d.shape[-1])
				break
	print('d.reshape: ',d.shape)
	return d

'''处理每个路径下的文件，得到每个文件的降水或地形数据,其中地形和降水标签需要缩放'''
def progress(path):
	#获取数据
	#type = path.split('/')[-2]
	#print(type)
	print(path)
	with nc.Dataset(path) as data:
		'''分类型处理'''
		#地形
		if "dem" in path:
			d =data.variables['dem'][:]
		#降水输入
		elif "PRE" in path:
			d = data.variables['PRE10m'][:]  # squeeze()为numpy的函数，去除维度为1的维
		elif "PRS" in path:
			d = data.variables['PRS'][:]
		elif "RHU" in path:
			d = data.variables['RHU'][:]
		elif "TEM" in path:
			d = data.variables['TEM'][:]
		elif "WIND" in path:
			d = data.variables['WINDAvg2mi'][:]
		elif "WINS" in path:
			d = data.variables['WINSAvg2mi'][:]
		else:
			print("file-{} is useless!".format(path))
			d = np.array([])
		# 将不合理的数值置零。np.where(condition, x, y), 满足condition输出x,否则输出y
		d = np.where(np.isnan(d), 0, d)
		d = np.where(d < 0, 0, d)
		d = np.where(d > 9999, 0, d)
		print(d.shape, d.max())
		return d

def read_dem(input_file):
	with nc.Dataset(input_file) as data:
		if "dem" in input_file:
			d = np.array(data.variables['dem'][:])
			d = d.reshape(-1, d.shape[-2], d.shape[-1])
			return d
	return None

def read_factors(input_dir):
	path_list = prepare_data(input_dir)
	# [降水数据, 湿度数据]
	result = [[], []]
	for path in path_list:
		with nc.Dataset(path) as data:
			'''分类型处理'''
			if "PRE" in path:
				d = data.variables['PRE10m'][:]
				result[0].extend(d)
			# elif "PRS" in path:
			# 	d = data.variables['PRS'][:]
			# 	result[1].extend(d)
			elif "RHU" in path:
				d = data.variables['RHU'][:]
				result[1].extend(d)
			# elif "TEM" in path:
			# 	d = data.variables['TEM'][:]
			# 	result[3].extend(d)
			# elif "WIND" in path:
			# 	d = data.variables['WINDAvg2mi'][:]
			# 	result[4].extend(d)
			# elif "WINS" in path:
			# 	d = data.variables['WINSAvg2mi'][:]
			# 	result[5].extend(d)
			else:
				pass
	factors = np.array(result)
	# factors = np.where(np.isnan(factors), 0, factors)
	factors = np.where(factors < 0, 0, factors)
	factors = np.where(factors > 9999, 0, factors)

	print(factors.shape, factors[0].max(), factors[1].max())
	return factors


if __name__=='__main__':
	# progress('./label/4/RGF_2017060100_SCN_PRE10m.nc')
	read_factors('factors/4')
