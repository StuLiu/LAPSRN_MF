import tensorflow as tf
import numpy as np
import os
from read_nc import *
from utils import *
import sys
from write_nc import *

class LapSRN(object):
	def __init__(self, sess, Epoch, lr, dr, input_size_w, input_size_j, output_size_w, output_size_j, ckpt,  batch_size, is_train):
		self.sess = sess
		self.Epoch = Epoch
		self.lr, self.dr = lr, dr   # 学习率和递减率
		self.input_size_w, self.input_size_j,self.output_size_w, self.output_size_j= input_size_w,input_size_j,output_size_w, output_size_j
		self.ckpt = ckpt
		self.batch_size = batch_size
		self.is_train = is_train
		self.keep_prob = tf.placeholder(tf.float32)
		self.build_model()

	def build_model(self):
		self.global_step = tf.Variable(tf.constant(0))
		# tf.train.exponential_decay：使学习率指数衰减
		self.learn_rate = tf.train.exponential_decay(self.lr, self.global_step, self.Epoch // 10, self.dr)
		with tf.variable_scope('input'):
			# 输入低分辨率数据
			self.X = tf.placeholder(tf.float32, [self.batch_size, self.input_size_w, self.input_size_j, 3], name='x')
			# 输入低分辨率数据对应的高分辨率标签数据
			self.Y = tf.placeholder(tf.float32, [self.batch_size, self.output_size_w, self.output_size_j, 1], name='y')
			# self.w : 超分辨率分支反卷积权重
			self.w = tf.get_variable(shape=[4, 4, 1, 3], name='w', initializer=tf.ones_initializer())
			# w_input : 特征提取分支上第一层卷积时候使用的权重，卷积核第三个参数对应X的深度
			w_input = tf.get_variable(shape=[3, 3, 3, 32],
			                          initializer=tf.random_normal_initializer(stddev=np.sqrt(2. / (3 * 3 * 32))),
			                          name='w_input')
		with tf.variable_scope('weights'):
			# 特征提取模块，连续五层卷积+一层反卷积+一层卷积
			for i in range(5):
				w = tf.get_variable(shape=[3, 3, 32, 32],
				                    initializer=tf.random_normal_initializer(stddev=np.sqrt(2. / (3 * 3 * 32))),
				                    name='w' + str(i))
			# self.w_是一系列卷积后再进行反卷积使用的卷积核  4 4是卷积核的尺寸，1是输出通道数，32是输入通道数
			self.w_ = tf.get_variable(shape=[4, 4, 1, 32], initializer=tf.ones_initializer(), name='w_')
			# w_out是进行先卷积再反卷积之后再SAME卷积作为输出时使用的卷积核
			w_out = tf.get_variable(shape=[3, 3, 1, 1],
			                        initializer=tf.random_normal_initializer(stddev=np.sqrt(2. / (3 * 3 * 1))),
			                        name='w_out')

		# tf.nn.conv2d_transpose一共有5个参数：分别为：1、指需要做反卷积的输入图像，它要求是一个Tensor
		# 2、卷积核，它要求是一个Tensor，具有[filter_height, filter_width, out_channels, in_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，卷积核个数，图像通道数]
		# 3、反卷积操作输出的shape。4、反卷积时在图像每一维的步长，这是一个一维的向量，长度4。5、string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
		##-----------------------------------------------------------------------------------##
		# 超分辨率分支：使用反卷积把输入的数据提高到label大小，此时以变成单通道。并使用Leaky_relu激活函数进行激活
		self.I = tf.nn.conv2d_transpose(self.X, self.w,
		                                output_shape=[self.batch_size, self.output_size_w, self.output_size_j, 1],
		                                strides=[1, 5, 5, 1], padding='SAME')
		self.I = tf.maximum(0.1 * self.I, self.I)  # realize the leaky_relu function

		# 特征提取分支第一步：使用w_input对输入进行SAME卷积，并使用Leaky_relu激活函数进行激活
		conv_input = tf.nn.conv2d(self.X, w_input, strides=[1, 1, 1, 1], padding='SAME')
		conv_input = tf.maximum(0.1 * conv_input, conv_input)
		# 特征提取分支第二步：五层SAME卷积
		with tf.variable_scope('weights', reuse=True):
			for i in range(5):
				w = tf.get_variable('w' + str(i))
				conv_input = tf.nn.conv2d(conv_input, w, strides=[1, 1, 1, 1], padding='SAME')
				conv_input = tf.maximum(conv_input * 0.2, conv_input, name='conv' + str(i))
		# 特征提取分支第三步：使用反卷积操作，对conv_input进行上采样到output_size大小。这里使用的权重为self.w_，使用Leaky_relu激活函数进行激活
		conv_10 = tf.nn.conv2d_transpose(conv_input, self.w_,
		                                 output_shape=[self.batch_size, self.output_size_w, self.output_size_j, 1],
		                                 strides=[1, 5, 5, 1], padding='SAME')
		conv_10 = tf.maximum(conv_10 * 0.1, conv_10)
		# 特征提取分支第四步：反卷积之后，再次进行SAME卷积，作为特征提取分支的输出。
		self.R = tf.nn.conv2d(conv_10, w_out, strides=[1, 1, 1, 1], padding='SAME')
		self.R = tf.maximum(self.R * 0.1, self.R)

		# 合并操作：将两个分支的输出相加，作为最终的输出。
		self.O = self.I + self.R
		self.O = tf.nn.dropout(self.O, self.keep_prob)

		# 分别计算：损失、学习到的残差、真实的残差
		self.loss = tf.reduce_mean(tf.sqrt(tf.square(self.Y - self.O) + 1e-6))
		self.rl = tf.reduce_mean(tf.abs(self.R) + 1e-6)
		self.rr = tf.reduce_mean(tf.abs(self.Y - self.I) + 1e-6)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.saver = tf.train.Saver()

	def train(self, config):
		input,label = preprocess(config)
		print("input's shape is:{0}, label's shape is:{1}".format(input.shape, label.shape))
		counter = 0
		self.sess.run(tf.global_variables_initializer())

		if self.load(self.ckpt):
			print('[*]LOADING checkpoint SUCCESS!')
		else:
			print('[!]LOADING checkpoint failed!')

		for ep in range(self.Epoch):
			idx = len(input)//self.batch_size
			np.random.seed(ep)
			np.random.shuffle(input)
			np.random.seed(ep)
			np.random.shuffle(label)
			for i in range(idx):
				counter += 1
				rate,loss,rr,rl,_ = self.sess.run([self.learn_rate,self.loss,self.rr,self.rl,self.optimizer],feed_dict={self.global_step:ep,self.X:input[i*self.batch_size:(i+1)*self.batch_size],self.Y:label[i*self.batch_size:(i+1)*self.batch_size],self.keep_prob:1})
				if counter % 10 == 0:
					print('Epoch: {0} lr: {1:.4} loss: {2:.4} res_learned: {3:.4} res_real: {4:.4}'.format(ep+1,rate,loss,rl,rr))
				if counter%200 == 0:
					self.save(self.ckpt,counter)

	def test(self, config):
		# input = preprocess_reconstruct(config)
		input = np.ones(shape=(287, 241, 401, 3),dtype='float32')
		print("测试时候，输入数据的shape是：{0}".format(input.shape))
		all_dimen = []
		time_dimen = input.shape[0]
		lat, lon = config.output_size_w, config.output_size_j

		all_dimen.append(time_dimen)
		all_dimen.append(lat)
		all_dimen.append(lon)
		self.sess.run(tf.global_variables_initializer())

		if self.load(self.ckpt):
			print('[*]LOADING SUCCESS', self.ckpt)
		else:
			print('[!]LOADING FAILED', self.ckpt)
			exit(-1)

		i = 0
		while (i+1) * self.batch_size < time_dimen:
			curr_batch = input[i * self.batch_size:(i + 1) * self.batch_size]
			Output = self.sess.run([self.O], feed_dict={self.X : curr_batch, self.keep_prob : 1})
			print(np.array(Output).shape)
			i += 1
		Output = self.sess.run([self.O], feed_dict={self.X : input[i * self.batch_size:], self.keep_prob : 1})
		print(np.array(Output).shape)
		# idx = len(input)//self.batch_size
		# writedata = []
		# for i in range(idx):
		# 	if i% hours_len == 0:
		# 		print('Reconstructing   ' + names[(i//hours_len)] +':')
		#
		# 	Output = self.sess.run([self.O],feed_dict={self.X:input[i*self.batch_size:(i+1)*self.batch_size], self.keep_prob:1})
		# 	# w,w_,Input,Res,Output = self.sess.run([self.w,self.w_,self.I,self.R,self.O],feed_dict={self.X:input[i*self.batch_size:(i+1)*self.batch_size],self.keep_prob:1})
		# 	#print("The shape of OutPut is:{0}".format(np.array(Output).shape))
		# 	writedata.append(np.array(Output).squeeze())    # (时间维, 纬度维, 经度维)
		#
		#
		# 	print('{0:11}/{1}   |'.format(i%hours_len+1, str(hours_len))+'██'*(i%hours_len+1)+'  '*(hours_len-1-i%hours_len)+'|',end='\r')
		# 	sys.stdout.flush()
		#
		# 	if (i+1) % hours_len == 0:
		# 		outnc = [i for i in all_dimen]
		# 		#print(outnc[0])
		# 		outnc.append(writedata)
		# 		writepath = os.path.join(config.output_dir, names[(i+1)//hours_len-1])
		# 		writepath = os.path.join(os.getcwd(), writepath)
		# 		writenc(writepath, config.factor_str, outnc)
		#
		# 		print(np.array(writedata).shape)
		# 		plt.imsave('example/output_RHU_1.png', np.array(writedata)[0, :, :])
		# 		plt.imsave('example/output_RHU_2.png', np.array(writedata)[71, :, :])
		#
		# 		writedata = []
		# 		print('{0:11}/{1}   |'.format(i%hours_len+1, str(hours_len))+'██'*(i%hours_len+1)+'  '*(hours_len-1-i%hours_len)+'|   COMPLETE!')
		# 		sys.stdout.flush()


	def load(self, checkpoint_dir):
		model_dir = "%s_%s_%s" % ("lapsrn", self.output_size_w, self.output_size_j)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess,os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def save(self, checkpoint_dir, step):
		model_name = "LAPSRN.model"
		model_dir = "%s_%s_%s" % ("lapsrn", self.output_size_w, self.output_size_j)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
			os.path.join(checkpoint_dir, model_name),
			global_step=step)
