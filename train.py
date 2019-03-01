from lapsrn import LapSRN
import pprint
import tensorflow as tf
import os
from utils import parse_config_test
import sys

mode_str ,area_str, factor_str = None, None, None
try:
	mode_str = sys.argv[1].split("=")[1].upper()    # NMC or others
	area_str = sys.argv[2].split("=")[1]            # 1, 2, 3, 4, 5, 6, 7, 8
	factor_str = sys.argv[3].split("=")[1].upper()  # PRE, PRS, RHU, TEM, WIND, WINS
except Exception:
	print("train.py接收参数错误，类似：python train.py mode=NMC area=1 factor=PRE")
	exit()

# 获取配置文件路径
config_path = os.path.join(os.getcwd(), "config")
config_path = os.path.join(config_path, mode_str + ".conf")

flags = tf.app.flags
FLAGS = flags.FLAGS

print("配置文件路径为:{0}，是否存在：{1}".format(config_path, os.path.exists(config_path)))
try:
	Epoch = int(parse_config_test(config_path, "common_option", "Epoch"))
	learning_rate = float(parse_config_test(config_path, "common_option", "learning_rate"))
	decay_rate = float(parse_config_test(config_path, "common_option", "dr"))
	batch_size = int(parse_config_test(config_path, "common_option", "batch_size"))
	factors_dir = os.path.join("factors", area_str)     # 气象要素数据集文件目录
	factor_str = factor_str     # 需要降尺度的气象要素代码: PRE, PRS, RHU, TEM, WIND, WINS
	dem_dir = os.path.join("dem", area_str)
	checkpoint = os.path.join("checkpoint", mode_str)
	checkpoint = os.path.join(checkpoint, area_str)
	checkpoint = os.path.join(checkpoint, factor_str)
	input_size_w = int(parse_config_test(config_path, area_str, "input_size_w"))
	input_size_j = int(parse_config_test(config_path, area_str, "input_size_j"))
	output_size_w = int(parse_config_test(config_path, area_str, "output_size_w"))
	output_size_j = int(parse_config_test(config_path, area_str, "output_size_j"))

	flags.DEFINE_integer('Epoch', Epoch, 'the number of epochs')
	flags.DEFINE_float('lr', learning_rate, 'learning rate')
	flags.DEFINE_float('dr', decay_rate, 'decay rate')
	flags.DEFINE_integer('input_size_w', input_size_w, 'input size of image')
	flags.DEFINE_integer('input_size_j', input_size_j, 'input size of image')
	flags.DEFINE_integer('output_size_w', output_size_w, 'output size of image')
	flags.DEFINE_integer('output_size_j', output_size_j, 'output size of image')
	flags.DEFINE_string('ckpt', checkpoint, 'directory of checkpoint')
	flags.DEFINE_string('dem_dir', dem_dir, 'directory of dem')
	flags.DEFINE_string('factors_dir', factors_dir, 'directory of label')
	flags.DEFINE_string('factor_str', factor_str, 'directory of label')

	flags.DEFINE_integer('batch_size', batch_size, 'batch size')
	flags.DEFINE_boolean('is_train', True, 'True for training False for test')
except Exception as e:
	print("获取配置信息时出错：", e)


from utils import preprocess
def main(_):
	pp = pprint.PrettyPrinter()
	pp.pprint(FLAGS.__flags)
	if not os.path.exists(FLAGS.ckpt):
		os.makedirs(FLAGS.ckpt)

	with tf.Session() as sess:
		# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
		# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		lapsrn = LapSRN(sess,
			FLAGS.Epoch,
			FLAGS.lr,
			FLAGS.dr,
			FLAGS.input_size_w,
			FLAGS.input_size_j,
			FLAGS.output_size_w,
			FLAGS.output_size_j,
			FLAGS.ckpt,
			FLAGS.batch_size,
			FLAGS.label,
			FLAGS.is_train
		)
		lapsrn.train(FLAGS)


if __name__ == '__main__':
	tf.app.run()
