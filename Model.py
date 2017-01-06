from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

class Model():
	def __init__(self, sess, X_train, y_train, batch_size=BATCH_SIZE):
		self.sess = sess
		self.X = tf.placeholder(tf.float32, shape=[None, 480, 640, 3], name="X")
		self.y = tf.placeholder(tf.float32, shape=[None], name="y")
		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
		self.phase_train = tf.placeholder(tf.bool, name="phase_train")
		
		self.prediction = inference(self.X, self.keep_prob)
		self.loss = loss(self.prediction, self.y)
		self.train_op = train(self.loss)
		
		self.X_train = X_train
		self.y_train = y_train
		
		self.saver = tf.train.Saver()
		self.writer = tf.train.SummaryWriter("log", graph=self.sess.graph)
		self.summary_op = tf.merge_all_summaries()
		self.sess.run(tf.initialize_all_variables())

	def train(self):
		for epoch in range(10000):
			#while l > .5 and epoch < 1500:
			offset = 0
			l = 0
			while offset < len(self.X_train):
				#l = 200
				#epoch = 0
				X_batch, y_batch = load_minibatch(self.X_train, self.y_train, offset)
				#while l > .05 and epoch < 5000:
				# Train
				#test_v = [v for v in tf.all_variables()if v.name == "conv1_weights:0"]
				#print(test_v)
				#test_v = test_v[0]
				#a = self.sess.run(test_v)
				_, l, summary = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict={self.X: X_batch, self.y: y_batch, self.keep_prob: 0.1})
				#b = self.sess.run(test_v)
				#print(np.array_equal(a, b))
				#print(test)
				#print("MiniBatch loss:", l)
				
				self.writer.add_summary(summary, offset)
				#print("Finished minibatch with offset", offset)
				# Update
				offset = min(offset + BATCH_SIZE, len(self.X_train))
			#if (epoch) % 50 == 0:
			print("Epoch:", epoch, "Loss:", l)
			self.saver.save(self.sess, "model.pkl")

		#test = self.sess.run([self.prediction], feed_dict={self.X: X_batch, self.keep_prob:1.0})
		#print(test)
		print("Optimization Finished!")
		self.saver.save(self.sess, "model.pkl")

	def load(self, save_path):
		self.saver.restore(self.sess, save_path)

	def predict(self, image):
		angle = self.sess.run(self.prediction, feed_dict={self.X: image, self.keep_prob: 1.0})
		return angle
		
		
# TODO
# Add Dropout
# Rename stuff
# Add Scopes
# Divide res layers
# Add res layers
# initialize to better weights?
# add more FC layers to reduce final layer fan in
# Change Conv -> FC change to use convolutions
# Added variables to FC lates
# Use leaky relu

# Use batch norm
# Add LSTM or Gated Conv to the Fully connected parts
# Predict more than steering angle, throttle and brake?
# Use a small set of images to predict, like last 5 - 3D convolution

def inference(images, keep_prob):
	INPUT_DEPTH = 3
	DEPTH1 = 16
	DEPTH2 = 32
	DEPTH3 = 64
	DEPTH4 = 128
	DEPTH5 = 256
	DEPTH6 = 512
	OUTPUT_DEPTH = 1
	# 7x7 - 16
	weightsc1 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[7, 7, INPUT_DEPTH, DEPTH1], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='conv1_weights') 
	biasesc1 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH1]), 
						name='conv1_biases')
	step1 = tf.nn.relu(conv2d(images, weightsc1) + biasesc1)
	# Pooling 
	step2 = max_pool_2x2(step1, 1)
	# Res 3x3 -16
	res = step2
	weightsres1 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH1, DEPTH1], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res1_weights') 
	biasesres1 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH1]), 
						name='res1_biases')
	step = tf.nn.relu(conv2d(step2, weightsres1, type="SAME") + biasesres1)

	weightsres2 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH1, DEPTH1], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res2_weights') 
	biasesres2 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH1]), 
						name='res2_biases')
	step = tf.nn.relu(conv2d(step, weightsres2, type="SAME") + biasesres2)

	step2 = tf.add(step, res)
	# step2 = tf.div(step2, 2.0)

	# 3x3 - 32
	weightsc2 = tf.get_variable(
						#shape=[3, 3, DEPTH1, DEPTH2],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH1, DEPTH2], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='conv2_weights') 
	biasesc2 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH2]), 
						name='conv2_biases')
	step3 = tf.nn.relu(conv2d(step2, weightsc2) + biasesc2)
	# pooling
	step4 = max_pool_2x2(step3, 3)
	#step4 = tf.nn.local_response_normalization(step4)
	# res 3x3 -32
	res = step4
	weightsres3 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH2, DEPTH2], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res3_weights') 
	biasesres3 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH2]), 
						name='res3_biases')
	step = tf.nn.relu(conv2d(step4, weightsres3, type="SAME") + biasesres3)

	weightsres4 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH2, DEPTH2], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res4_weights') 
	biasesres4 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH2]), 
						name='res4_biases')
	step = tf.nn.relu(conv2d(step, weightsres4, type="SAME") + biasesres4)

	step4 = tf.add(step, res)
	# Pooling 
	step5 = max_pool_2x2(step4, 4)
	# 3x3 - 64
	weightsc3 = tf.get_variable(
						#shape=[3, 3, DEPTH2, DEPTH3],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH2, DEPTH3], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='conv3_weights') 
	biasesc3 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH3]), 
						name='conv3_biases')
	step6 = tf.nn.relu(conv2d(step5, weightsc3) + biasesc3)
	# pooling
	step7 = max_pool_2x2(step6, 5)
	#step7 = tf.nn.local_response_normalization(step7)
	# Res 3x3
	res = step7
	weightsres5 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH3, DEPTH3], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res5_weights') 
	biasesres5 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH3]), 
						name='res5_biases')
	step = tf.nn.relu(conv2d(step7, weightsres5, type="SAME") + biasesres5)

	weightsres6 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH3, DEPTH3], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res6_weights') 
	biasesres6 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH3]), 
						name='res6_biases')
	step = tf.nn.relu(conv2d(step, weightsres6, type="SAME") + biasesres6)

	step7 = tf.add(step, res)
	# 3x3 - 128
	weightsc4 = tf.get_variable(
						#shape=[3, 3, DEPTH3, DEPTH4],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH3, DEPTH4], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='conv4_weights') 
	biasesc4 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH4]), 
						name='conv4_biases')
	step8 = tf.nn.relu(conv2d(step7, weightsc4) + biasesc4)
	# pooling
	step9 = max_pool_2x2(step8, 6)
	#step9 = tf.nn.local_response_normalization(step9)
	# res 3x3
	res = step9
	weightsres7 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH4, DEPTH4], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res7_weights') 
	biasesres7 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH4]), 
						name='res7_biases')
	step = tf.nn.relu(conv2d(step9, weightsres7, type="SAME") + biasesres7)

	weightsres8 = tf.get_variable(
						#shape=[7, 7, INPUT_DEPTH, DEPTH1],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH4, DEPTH4], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='res8_weights') 
	biasesres8 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH4]), 
						name='res8_biases')
	step = tf.nn.relu(conv2d(step, weightsres8, type="SAME") + biasesres8)

	step9 = tf.add(step, res)
	# 3x3 - 256
	weightsc5 = tf.get_variable(
						#shape=[3, 3, DEPTH4, DEPTH5],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH4, DEPTH5], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='conv5_weights') 
	biasesc5 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH5]), 
						name='conv5_biases')
	step10 = tf.nn.relu(conv2d(step9, weightsc5) + biasesc5)
	# pooling
	step11 = max_pool_2x2(step10, 7)
	#step11 = tf.nn.local_response_normalization(step11)
	# 3x3 - 512
	weightsc6 = tf.get_variable(
						#shape=[3, 3, DEPTH5, DEPTH6],
						initializer=tf.truncated_normal(shape=[3, 3, DEPTH5, DEPTH6], stddev=0.1),
						#initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='conv6_weights') 
	biasesc6 = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH6]), 
						name='conv6_biases')
	step12 = tf.nn.relu(conv2d(step11, weightsc6) + biasesc6)
	# pooling
	step13 = max_pool_2x2(step12, 8)
	step13 = max_pool_2x2(step13, 2)
	#step13 = tf.nn.local_response_normalization(step13)
	# Flatten
	shape = step13.get_shape().as_list()
	dim = np.prod(shape[1:])
	print("####################",dim,"########################")
	flatten = tf.reshape(step13, [-1, dim])

	# fc1
	weightsfc1 = tf.get_variable(initializer=tf.truncated_normal(stddev=0.1, shape=[dim, 4096]), name="fc1_weights")
	biasesfc1 = tf.get_variable(initializer=tf.constant(1.0, shape=[4096]), name="fc1_bias")
	step14 = tf.nn.relu(tf.add(tf.matmul(flatten, weightsfc1), biasesfc1))
	
	#fc2
	weightsfc2 = tf.get_variable(initializer=tf.truncated_normal(stddev=0.1, shape=[4096, 4096]), name="fc2_weights")
	biasesfc2 = tf.get_variable(initializer=tf.constant(1.0, shape=[4096]), name="fc2_bias")
	step15 = tf.nn.relu(tf.add(tf.matmul(step14, weightsfc2), biasesfc2))

	#fc3
	weightsfc3 = tf.get_variable(initializer=tf.truncated_normal(stddev=0.1, shape=[4096, 1024]), name="fc3_weights")
	biasesfc3 = tf.get_variable(initializer=tf.constant(1.0, shape=[1024]), name="fc3_bias")
	step16 = tf.nn.relu(tf.add(tf.matmul(step15, weightsfc3), biasesfc3))

	weightsfc4 = tf.get_variable(initializer=tf.truncated_normal(stddev=0.1, shape=[1024, 1024]), name="fc4_weights")
	biasesfc4 = tf.get_variable(initializer=tf.constant(1.0, shape=[1024]), name="fc4_bias")
	step17 = tf.nn.relu(tf.add(tf.matmul(step16, weightsfc4), biasesfc4))

	#output
	weights = tf.get_variable(initializer=tf.truncated_normal(stddev=0.1, shape=[1024, 1]), name="output_weights")
	biases = tf.get_variable(initializer=tf.constant(1.0, shape=[1]), name="output_bias")
	#return tf.mul(tf.atan(tf.matmul(step17, weights) + biases), 2)
	return tf.matmul(step17, weights) + biases

def loss(prediction, y):
	#print(tf.reshape(prediction, [-1]), y)
	loss = tf.reduce_sum(tf.pow((tf.transpose(tf.reshape(prediction, [-1])) - y), 2))
	tf.scalar_summary("loss", loss)
	return loss

def train(total_loss):
	# global_step = tf.Variable(0)
	# learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
	# return tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
	return tf.train.AdamOptimizer(1e-4).minimize(total_loss)