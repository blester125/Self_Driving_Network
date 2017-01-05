from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

# This function will be useful when I get Batch normalization working
def convolution2D(
		x, 
		output_depth,
		kernel_size,
		strides=[1, 1], 
		padding='SAME',
		name="convlution",
		weight_decay=0.0,
		stddev=1e-1
		):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[3]
		regularizer = lambda t: l2_loss(t, weight=weight_decay)
		kernel = tf.get_variable(
			"weights",
			[kernel_size[0],
			 kernel_size[1],
			 input_depth,
			 output_depth],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			regularizer=regularizer,
			dtype=x.dtype)
		conv = tf.nn.conv2d(x, kernel, [1, strides[0], strides[1], 1], padding=padding)
	return conv

def convolution_and_bias(
		x, 
		output_depth,
		kernel_size,
		strides=[1, 1], 
		padding='SAME',
		name="convlution",
		weight_decay=0.0,
		stddev=1e-1
		):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[3]
		regularizer = lambda t: l2_loss(t, weight=weight_decay)
		kernel = tf.get_variable(
			"weights",
			[kernel_size[0],
			 kernel_size[1],
			 input_depth,
			 output_depth],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			regularizer=regularizer,
			dtype=x.dtype)
		biases = tf.get_variable(
			initializer=tf.constant(0.1, shape=[output_depth]), 
			name="biases", 
			dtype=x.dtype)
		conv = tf.nn.conv2d(x, kernel, [1, strides[0], strides[1], 1], padding=padding)
		activation = conv + biases
	return activation

# Still working to get batchnorm to work well
# def convolution2D(
# 		x, 
# 		output_depth,
# 		kernel_size,
# 		strides=[1, 1], 
# 		padding='SAME',
# 		name="convlution",
# 		weight_decay=0.0,
# 		stddev=1e-1
# 		):
# 	with tf.variable_scope(name):
# 		conv = tf.contrib.layers.convolution2d(
# 			inputs=x,
# 			num_outputs=output_depth,
# 			kernel_size=kernel_size,
# 			stride=strides,
# 			padding=padding,
# 			rate=1,
# 			activation_fn=None,
# 			normalizer_fn=tf.contrib.layers.batch_norm,
# 			normalizer_params={'decay':0.9, 'center': True, 'scale': True, 'updates_collections':None},
# 			weights_initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
# 			biases_initializer=tf.zeros_initializer,
# 			trainable=True,
# 			scope='cnn'
# 			)
# 	return conv

def fully_connected(x, output_depth, name="fully_connected", weight_decay=0.0, stddev=1e-1):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[-1]
		regularizer = lambda t: l2_loss(t, weight=weight_decay)
		weights = tf.get_variable(
			"weights",
			[input_depth, output_depth],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			regularizer=regularizer,
			dtype=x.dtype)
		biases = tf.get_variable(
			initializer=tf.constant(0.1, shape=[output_depth]), 
			name="biases", 
			dtype=x.dtype)
		activation = tf.matmul(x, weights) + biases
		out = tf.nn.relu(activation)
	return out

def softmax(x, output_depth, name="softmax", stddev=1e-1):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[-1]
		weights = tf.get_variable(
			"weights",
			[input_depth, output_depth],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			dtype=x.dtype)
		biases = tf.get_variable(
			initializer=tf.constant(0.1, shape=[output_depth]), 
			name="biases", 
			dtype=x.dtype)
		activation = tf.matmul(x, weights) + biases
		out = tf.nn.softmax(activation)
	return out

def linear_regression(x, name="Linear_Regression", stddev=1e-1):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[-1]
		weights = tf.get_variable(
			"weights",
			[input_depth, 1],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			dtype=x.dtype)
		biases = tf.get_variable(
			initializer=tf.constant(0.1, shape=[output_depth]), 
			name="biases", 
			dtype=x.dtype)
		activation = tf.matmul(x, weights) + biases
	return activation

def l2_loss(tensor, weight=1.0, name=None):
	with tf.name_scope(name):
		weight = tf.convert_to_tensor(weight, dtype=tensor.dtype.base_dtype, name='loss_weight')
		loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
	return loss

def max_pool(
		x, 
		kernel=[2, 2], 
		strides=[2, 2], 
		padding='SAME', 
		name="Max_Pool"):
	with tf.variable_scope(name):
		maxpool = tf.nn.max_pool(
							x,
							ksize=[1, kernel[0], kernel[1], 1],
							strides=[1, strides[0], strides[1], 1],
							padding=padding)
	return maxpool

def min_pool(
		x, 
		kernel=[2, 2],
		strides=[2, 2], 
		padding='SAME', 
		name="Min_Pool"):
	with tf.variable_scope(name):
		minpool = tf.nn.min_pool(
							x,
							ksize=[1, kernel[0], kernel[1], 1],
							strides=[1, strides[0], strides[1], 1],
							padding=padding)
	return minpool

def avg_pool(
		x, 
		kernel=[2, 2],
		strides=[2, 2],
		padding='SAME', 
		name="Avg_Pool"):
	with tf.variable_scope(name):
		maxpool = tf.nn.avg_pool(
							x,
							ksize=[1, kernel[0], kernel[1], 1],
							strides=[1, strides[0], strides[1], 1],
							padding=padding)
	return avgpool

def lp_pool(
		x,
		p=2,
		kernel=[2, 2],
		strides=[2, 2],
		padding='SAME',
		name=None):
	if name is None:
		name = "L" + str(p) + "_Pool"
	with tf.variable_scope(name):
		power = tf.pow(x, p)
		subsample = tf.nn.avg_pool(
								power,
								k_size=[1, kernel[0], kernel[1], 1],
								strides=[1, strides[0], strides[1], 1],
								padding=padding)
		subsample_sum = tf.mul(subsample, k_height * k_width)
		out = tf.pow(subsample_sum, 1/p)
		return out

def leaky_relu(x, alpha=0.2, name="Leaky_ReLU"):
	return tf.maximum(alpha * x, x)

# def batch_norm(x, phase_train, epsilon=1e-3):
# 	with tf.variable_scope("Batch_Norm"):
# 		#phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
# 		n_out = int(x.get_shape()[3])
# 		print(n_out)
# 		beta = tf.Variable(
# 					tf.constant(0.0, shape=[n_out], dtype=x.dtype), 
# 					name="beta", 
# 					trainable=True, 
# 					dtype=x.dtype
# 				)
# 		gamma = tf.Variable(
# 					tf.constant(1.0, shape=[n_out], dtype=x.dtype), 
# 					name="gamma", 
# 					trainable=True, 
# 					dtype=x.dtype
# 				)
# 		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
# 		ema = tf.train.ExponentialMovingAverage(decay=0.5)
# 		def mean_var_with_update():
# 			ema_apply_op = ema.apply([batch_mean, batch_var])
# 			with tf.control_dependencies([ema_apply_op]):
# 				return tf.identity(batch_mean), tf.identity(batch_var)
# 		mean, var = control_flow_ops.cond(phase_train,
# 										  mean_var_with_update,
# 										  lambda: (ema.average(batch_mean), ema.average(batch_var)))
# 		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
# 	return normed

# def batch_norm_layer(x,train_phase,scope_bn):
# 	bn_train = batch_norm(
# 		x, 
# 		decay=0.9, 
# 		center=True, 
# 		scale=True,
# 		is_training=True,
# 		reuse=None,
# 		trainable=True,
# 		scope=scope_bn,
# 		updates_collections=None)

# 	bn_inference = batch_norm(
# 		x, 
# 		decay=0.9, 
# 		center=True, 
# 		scale=True,
# 		is_training=False,
# 		reuse=True,
# 		trainable=True,
# 		scope=scope_bn,
# 		updates_collections=None)

# 	z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
# 	return z

if __name__ == "__main__":
	sess = tf.InteractiveSession()
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	phase_train = tf.placeholder(tf.bool, name='phase_train')
	keep_prob = tf.placeholder(tf.float32)

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	conv1 = convolution_and_bias(x_image, 32, [5, 5], name="c1")
	h_conv1 = tf.nn.relu(conv1)

	h_pool1 = max_pool(h_conv1)

	conv2 = convolution_and_bias(h_pool1, 64, [5, 5], name="c2")
	h_conv2 = tf.nn.relu(conv2)

	h_pool2 = max_pool(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

	fc1 = fully_connected(h_pool2_flat, 1024, "fc1")
	h_fc1_drop = tf.nn.dropout(fc1, keep_prob)

	y_conv = softmax(h_fc1_drop, 10, name="softmax")

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	init_op = tf.initialize_all_variables()

	sess.run(init_op)
	for i in range(5000):
		batch = mnist.train.next_batch(100)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(
								feed_dict={
									x:batch[0], 
									y_:batch[1], 
									keep_prob:1.0, 
									phase_train:False})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={
							x:batch[0], 
							y_:batch[1], 
							keep_prob:0.5, 
							phase_train:True})

	print("Test Accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0, phase_train:False}))