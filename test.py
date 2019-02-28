from lapsrn import LapSRN
import pprint
import tensorflow as tf
import os
from utils import parse_config_test
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def run_test(config_path,test_set_path, area_str, mode_str, output_dir):
  try:
    input_ = test_set_path
    output_ = output_dir
    label = os.path.join("label", area_str)
    feature = os.path.join("dem", area_str)
    checkpoint = os.path.join("checkpoint", mode_str)
    checkpoint = os.path.join(checkpoint, area_str)
    input_size_w = int(parse_config_test(config_path, area_str, "input_size_w"))
    input_size_j = int(parse_config_test(config_path, area_str, "input_size_j"))
    output_size_w = int(parse_config_test(config_path, area_str, "output_size_w"))
    output_size_j = int(parse_config_test(config_path, area_str, "output_size_j"))
    Epoch = int(parse_config_test(config_path, "common_option", "Epoch"))
    learning_rate = float(parse_config_test(config_path, "common_option", "learning_rate"))
    decay_rate = float(parse_config_test(config_path, "common_option", "dr"))
    
  except Exception as e:
    print("获取配置信息时出错：", e)

  flags = tf.app.flags

  flags.DEFINE_integer('Epoch',Epoch,'the number of epochs')
  flags.DEFINE_float('lr', learning_rate, 'learning rate')
  flags.DEFINE_float('dr', decay_rate, 'decay rate')
  flags.DEFINE_string('input', input_, 'directory of input')
  flags.DEFINE_string('output', output_, 'directory of output')
  flags.DEFINE_string('label', label, 'directory of label')
  flags.DEFINE_string('feature', feature, 'directory of dem')


  flags.DEFINE_integer('input_size_w', input_size_w, 'input size of image')
  flags.DEFINE_integer('input_size_j', input_size_j, 'input size of image')

  flags.DEFINE_integer('output_size_w', output_size_w, 'output size of image')
  flags.DEFINE_integer('output_size_j', output_size_j, 'output size of image')
  flags.DEFINE_integer('batch_size',1,'batch size')
  flags.DEFINE_string('ckpt', checkpoint, 'directory of checkpoint')
  flags.DEFINE_boolean('is_train',False,'True for training False for test')
  FLAGS = flags.FLAGS

  pp = pprint.PrettyPrinter()
  FLAGS._parse_flags()
  pp.pprint(flags.FLAGS.__flags)
  if not os.path.exists(FLAGS.output):
    os.makedirs(FLAGS.output)

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
        lapsrn.test(FLAGS)

# if __name__ == '__main__':
	# tf.app.run()
