from lapsrn import LapSRN
import pprint
import tensorflow as tf
import os
from utils import parse_config_test
import sys

try:
  mode_str = sys.argv[1].split("=")[1].upper()
  area_str = sys.argv[2].split("=")[1]
except Exception:
  print("train.py接收参数错误，类似：python train.py mdoe=NMC area=1")
  exit()

# 获取配置文件路径
config_path = os.path.join(os.getcwd(), "config")
config_path = os.path.join(config_path, mode_str + ".conf")

print("配置文件路径为:{0}，是否存在：{1}".format(config_path, os.path.exists(config_path)))
try:
  Epoch = int(parse_config_test(config_path, "common_option", "Epoch"))
  learning_rate = float(parse_config_test(config_path, "common_option", "learning_rate"))
  decay_rate = float(parse_config_test(config_path, "common_option", "dr"))
  batch_size = int(parse_config_test(config_path, "common_option", "batch_size"))
  label = os.path.join("label", area_str)
  feature = os.path.join("dem", area_str)
  checkpoint = os.path.join("checkpoint", mode_str)
  checkpoint = os.path.join(checkpoint, area_str)
  input_size_w = int(parse_config_test(config_path, area_str, "input_size_w"))
  input_size_j = int(parse_config_test(config_path, area_str, "input_size_j"))
  output_size_w = int(parse_config_test(config_path, area_str, "output_size_w"))
  output_size_j = int(parse_config_test(config_path, area_str, "output_size_j"))
except Exception as e:
  print("获取配置信息时出错：", e)


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('Epoch',Epoch,'the number of epochs')
flags.DEFINE_float('lr', learning_rate, 'learning rate')
flags.DEFINE_float('dr', decay_rate, 'decay rate')
flags.DEFINE_integer('input_size_w', input_size_w, 'input size of image')
flags.DEFINE_integer('input_size_j', input_size_j, 'input size of image')


flags.DEFINE_integer('output_size_w', output_size_w, 'output size of image')
flags.DEFINE_integer('output_size_j', output_size_j, 'output size of image')

flags.DEFINE_string('ckpt', checkpoint, 'directory of checkpoint')

flags.DEFINE_string('feature', feature, 'directory of dem')
flags.DEFINE_integer('batch_size',batch_size,'batch size')
flags.DEFINE_string('label', label, 'directory of label')
flags.DEFINE_boolean('is_train',True,'True for training False for test')

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
						FLAGS.is_train)
		lapsrn.train(FLAGS)

if __name__ == '__main__':
	tf.app.run()
