from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
import copy

from network import *

BATCH_SIZE = 12

class Model():
	def __init__(self, sess, X_train, y_train, batch_size=BATCH_SIZE):
		self.sess = sess
		self.X = tf.placeholder(tf.float32, shape=[None, 480, 640, 3], name="X")
		self.y = tf.placeholder(tf.float32, shape=[None], name="y")
		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
		self.phase_train = tf.placeholder(tf.bool, name="phase_train")
		
		self.prediction = inference(self.X, self.keep_prob, self.phase_train)
		self.loss = loss(self.prediction, self.y)
		self.train_op = train(self.loss)
		
		self.X_train = X_train
		self.y_train = y_train
		
		self.saver = tf.train.Saver()
		self.writer = tf.train.SummaryWriter("log", graph=self.sess.graph)
		self.summary_op = tf.merge_all_summaries()
		self.sess.run(tf.initialize_all_variables())

	def train(self):
		for epoch in range(1000):
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
				_, l, summary = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict={self.X: X_batch, self.y: y_batch, self.keep_prob: 0.1, self.phase_train: True})
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

def inference(images, keep_prob, phase_train):
	INPUT_DEPTH = 3
	DEPTH1 = 16
	DEPTH2 = 32
	DEPTH3 = 64
	DEPTH4 = 128
	DEPTH5 = 256
	DEPTH6 = 512
	OUTPUT_DEPTH = 1
	
	conv1 = convolution2D(images, DEPTH1, [7, 7], phase_train=phase_train, use_batch_norm=True, name="Conv1")
	conv1_o = tf.nn.relu(conv1)

	res1 = residual_block(conv1_o, DEPTH1, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Res1")
	res1_o = tf.nn.relu(res1)

	conv2 = convolution2D(res1_o, DEPTH2, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Conv2")
	conv2_o = tf.nn.relu(conv2)

	pool1 = max_pool(conv2_o, name="Pool1")

	res2 = residual_block(pool1, DEPTH2, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Res2")
	res2_o = tf.nn.relu(res2)

	pool2 = max_pool(res2_o, name="Pool2")

	conv3 = convolution2D(pool2, DEPTH3, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Conv3")
	conv3_o = tf.nn.relu(conv3)

	pool3 = max_pool(conv3_o, name="Pool3")

	res3 = residual_block(pool3, DEPTH3, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Res3")
	res3_o = tf.nn.relu(res3)

	conv4 = convolution2D(res3_o, DEPTH4, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Conv4")
	conv4_o = tf.nn.relu(conv4)

	pool4 = max_pool(conv4_o, name="Pool4")

	res4 = residual_block(pool4, DEPTH4, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Res4")
	res4_o = tf.nn.relu(res4)

	conv5 = convolution2D(res4_o, DEPTH5, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Conv5")
	conv5_o = tf.nn.relu(conv5)

	pool5 = max_pool(conv5_o, name="Pool5")

	conv6 = convolution2D(pool5, DEPTH6, [3, 3], phase_train=phase_train, use_batch_norm=True, name="Conv6")
	conv6_o = tf.nn.relu(conv6)

	pool6 = max_pool(conv6_o, name="Pool6")
	pool7 = max_pool(pool6, name="Pool7")
	pool8 = max_pool(pool7, name="Pool8")

	#Flatten
	shape = pool8.get_shape().as_list()
	print(shape)
	dim = np.prod(shape[1:])
	print("####################",dim,"########################")
	flatten = tf.reshape(pool8, [-1, dim])

	fc1 = fully_connected(flatten, 4096, name="FC1")
	dropout1 = tf.nn.dropout(fc1, keep_prob)

	fc2 = fully_connected(dropout1, 4096, name="FC2")
	dropout2 = tf.nn.dropout(fc2, keep_prob)

	fc3 = fully_connected(dropout2, 1024, name="FC3")
	dropout3 = tf.nn.dropout(fc3, keep_prob)

	fc4 = fully_connected(dropout3, 1024, name="FC4")
	dropout4 = tf.nn.dropout(fc4, keep_prob)

	output = linear_regression(dropout4, name="Output")
	return output
	

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

def load_minibatch(X_train, y_train, offset):
	y_batch = np.array(y_train[offset:min(offset + BATCH_SIZE, len(X_train))])
	X_batch = []
	for i in range(BATCH_SIZE):
		if offset + i >= len(X_train):
			break
		image = load_image(X_train[offset+i])
		X_batch.append(image)
	X_batch = np.array(X_batch)
	return X_batch, y_batch

def normalize_image(image):
	norm_image = copy.deepcopy(image)
	norm_image = cv2.normalize(
						image, 
						norm_image, 
						alpha=0, 
						beta=1, 
						norm_type=cv2.NORM_MINMAX, 
						dtype=cv2.CV_32F)
	return norm_image

def load_image(path, normalize=True):
	image = cv2.imread(path)
	if normalize:
		image = normalize_image(image)
	return image