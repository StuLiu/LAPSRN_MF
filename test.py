from lapsrn_MF_0 import LapSRN
import pprint
import tensorflow as tf
import os
from utils import parse_config_test
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def run_test(area_str, mode_str, factor_str, input_dir, output_dir):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    try:
        config_path = os.path.join('config', mode_str + '.conf')
        factors_dir = os.path.join("factors", area_str)
        dem_dir = os.path.join("dem", area_str)
        input_size_w = int(parse_config_test(config_path, area_str, "input_size_w"))
        input_size_j = int(parse_config_test(config_path, area_str, "input_size_j"))
        output_size_w = int(parse_config_test(config_path, area_str, "output_size_w"))
        output_size_j = int(parse_config_test(config_path, area_str, "output_size_j"))
        Epoch = int(parse_config_test(config_path, "common_option", "Epoch"))
        learning_rate = float(parse_config_test(config_path, "common_option", "learning_rate"))
        decay_rate = float(parse_config_test(config_path, "common_option", "dr"))
        # checkpoint/NMC/4/PRE/lapsrn_1201_2001
        checkpoint_dir = os.path.join("checkpoint", mode_str)
        checkpoint_dir = os.path.join(checkpoint_dir, area_str)
        checkpoint_dir = os.path.join(checkpoint_dir, factor_str)

        flags.DEFINE_integer('Epoch', Epoch, 'the number of epochs')
        flags.DEFINE_float('lr', learning_rate, 'learning rate')
        flags.DEFINE_float('dr', decay_rate, 'decay rate')
        flags.DEFINE_string('input_dir', input_dir, 'directory of input')
        flags.DEFINE_string('output_dir', output_dir, 'directory of output')
        flags.DEFINE_string('factors_dir', factors_dir, 'directory of factors')
        flags.DEFINE_string('dem_dir', dem_dir, 'directory of dem')
        flags.DEFINE_integer('input_size_w', input_size_w, 'input size of image')
        flags.DEFINE_integer('input_size_j', input_size_j, 'input size of image')
        flags.DEFINE_integer('output_size_w', output_size_w, 'output size of image')
        flags.DEFINE_integer('output_size_j', output_size_j, 'output size of image')
        flags.DEFINE_integer('batch_size', 1, 'batch size')
        flags.DEFINE_string('ckpt', checkpoint_dir, 'directory of checkpoint')
        flags.DEFINE_boolean('is_train', False, 'True for training False for test')
        flags.DEFINE_string('factor_str', factor_str, 'factor name')

        pp = pprint.PrettyPrinter()
        FLAGS._parse_flags()
        pp.pprint(flags.FLAGS.__flags)
    except Exception as e:
        print("获取配置信息时出错：", e)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

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
                        FLAGS.is_train)
        lapsrn.test(FLAGS)

if __name__ == '__main__':
    run_test(area_str='4',
         mode_str='NMC',
         factor_str='PRS',
         input_dir='factors/4',
         output_dir='output_1km'
    )
