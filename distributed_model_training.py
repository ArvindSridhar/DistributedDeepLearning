from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import numpy as np
import pandas
import matplotlib.pyplot as plt

class distributed_model_training:

	def __init__(self):
		self.get_data()
		self.distribute_data()

	def tensorflow_test(self):
		hello = tf.constant('Hello, TensorFlow!')
		sess = tf.Session()
		print(sess.run(hello))

	def get_data(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = x_train.reshape(60000, 784)
		x_test = x_test.reshape(10000, 784)
		self.y_train = y_train.reshape(60000, 1)
		self.y_test = y_test.reshape(10000, 1)
		self.x_train = x_train.astype('float32')
		self.x_test = x_test.astype('float32')
		self.x_train /= 255
		self.x_test /= 255
		print(self.x_train.shape)
		print(self.y_train.shape)

	def distribute_data(self):
		#only deal with x_train and y_train
		# np.random.shuffle(self.x_train)
		self.segment_batches = {}
		for i in range(10):
			self.segment_batches["seg"+str(i)] = (self.x_train[6000*i:6000*i+6000], self.y_train[6000*i:6000*i+6000])

model_instance = distributed_model_training()
# model_instance.get_data()


# def set_hyperparams():
# 	batch_size = 128
# 	num_classes = 10
# 	epochs = 20