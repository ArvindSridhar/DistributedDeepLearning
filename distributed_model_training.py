from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model
import numpy as np
import pandas
import matplotlib.pyplot as plt
import math

class distributed_model_training:

	def __init__(self):
		self.num_classes = 10
		self.total_num_epochs = 5 #Can tune
		self.batch_size = 100
		self.num_segments = 10 #Can tune
		self.num_iters_on_segment = 10 #Can tune
		self.get_data()
		self.distribute_data()
		self.define_models()
		self.train_model_aggregate()

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
		# One-hot encode the y vectors
		self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
		self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

	def distribute_data(self):
		# Shuffle? np.random.shuffle(self.x_train)
		self.segment_batches = {}
		data_per_segment = int(math.floor(60000/self.num_segments))
		for i in range(self.num_segments):
			self.segment_batches["seg"+str(i)] = (self.x_train[data_per_segment*i:data_per_segment*i+data_per_segment],
												  self.y_train[data_per_segment*i:data_per_segment*i+data_per_segment])

	def define_models(self):
		self.segment_models = {}
		for i in range(self.num_segments):
			model = Sequential()
			model.add(Dense(512, activation='relu', input_shape=(784,)))
			model.add(Dropout(0.2))
			model.add(Dense(512, activation='relu'))
			model.add(Dropout(0.2))
			model.add(Dense(self.num_classes, activation='softmax'))
			self.segment_models["seg"+str(i)] = model

	def train_model_aggregate(self):
		# Training and evaluation loop
		for i in range(self.total_num_epochs):
			print("Grand Epoch:", i+1, "/", self.total_num_epochs)
			
			# Re-define the aggregate model (stored on the master node, and ultimately returned), also re-initialize its weights
			self.aggregate_model = Sequential()
			self.aggregate_model.add(Dense(512, activation='relu', input_shape=(784,)))
			self.aggregate_model.add(Dropout(0.2))
			self.aggregate_model.add(Dense(512, activation='relu'))
			self.aggregate_model.add(Dropout(0.2))
			self.aggregate_model.add(Dense(self.num_classes, activation='softmax'))

			# Train individual models for 5 epochs
			for segment in sorted(self.segment_models):
				print('Segment:', segment)
				(x_train_seg, y_train_seg) = self.segment_batches[segment]				
				self.segment_models[segment].compile(loss='categorical_crossentropy',
					optimizer=Adam(),
					metrics=['accuracy'])
				history = self.segment_models[segment].fit(x_train_seg, y_train_seg,
			        batch_size=self.batch_size,
			        epochs=self.num_iters_on_segment,
			        verbose=1,
			        validation_data=(self.x_test, self.y_test))

			# Average the weights of the trained models on the segments, add these weights to the aggregate model
			avg_weights = sum([np.array(self.segment_models[segment].get_weights()) for segment in self.segment_models])/self.num_segments
			self.aggregate_model.set_weights(avg_weights)

			# Compile aggregate model
			self.aggregate_model.compile(loss='categorical_crossentropy',
				optimizer=Adam(),
				metrics=['accuracy'])

			# Evaluate aggregate model on the test set
			score = self.aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
			print(score)

			# Redistribute the aggregate model to each segment for the next epoch of training
			for segment in self.segment_models:
				self.segment_models[segment] = clone_model(self.aggregate_model)

			#clustering of weights: local minima (throw away the rest of the clusters if one is clearly best or merge clusters and try to re-train)
			#have the same dataset on each segment
			#Run more iterations between merging

		# Conduct final testing of aggregate model
		train_score = self.aggregate_model.evaluate(self.x_train, self.y_train, verbose=1)
		test_score = self.aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
		print("Training accuracy:", train_score[1])
		print("Test accuracy:", test_score[1])

model_instance = distributed_model_training()
