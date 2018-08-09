from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.models import clone_model
import numpy as np
import pandas
import matplotlib.pyplot as plt

class distributed_model_training:

	def __init__(self):
		self.num_classes = 10
		self.total_num_epochs = 1 #to change
		self.batch_size = 100
		self.num_segments = 10
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
		#One-hot encode the y vectors
		self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
		self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

	def distribute_data(self):
		#only deal with x_train and y_train
		# np.random.shuffle(self.x_train)
		self.segment_batches = {}
		for i in range(self.num_segments):
			self.segment_batches["seg"+str(i)] = (self.x_train[6000*i:6000*i+6000], self.y_train[6000*i:6000*i+6000])

	def define_models(self):
		self.segment_models = {}
		for i in range(self.num_segments):
			model = Sequential()
			model.add(Dense(512, activation='relu', input_shape=(784,)))
			model.add(Dropout(0.2))
			model.add(Dense(512, activation='relu'))
			model.add(Dropout(0.2))
			model.add(Dense(self.num_classes, activation='softmax'))

			model.compile(loss='categorical_crossentropy',
	            optimizer=Adam(),
	            metrics=['accuracy'])

			self.segment_models["seg"+str(i)] = model

	def train_model_aggregate(self):
		for i in range(self.total_num_epochs):
			for segment in self.segment_models:
				# model_seg = self.segment_models[segment]
				(x_train_seg, y_train_seg) = self.segment_batches[segment]				
				# self.segment_models[segment] = self.train_model_segment(model_seg, x_train_seg, y_train_seg)
				history = self.segment_models[segment].fit(x_train_seg, y_train_seg,
			        batch_size=self.batch_size,
			        epochs=1,
			        verbose=1,
			        validation_data=(self.x_test, self.y_test))

		#--------------------------------------------------------------------------------

		for segment in self.segment_models:
			score = self.segment_models[segment].evaluate(self.x_test, self.y_test, verbose=0)
			print('[Segment ' + str(segment) + '] Test loss: ' + str(score[0]))
			print('[Segment ' + str(segment) + '] Test accuracy: ' + str(score[1]))

		#--------------------------------------------------------------------------------

		aggregate_model = Sequential()
		aggregate_model.add(Dense(512, activation='relu', input_shape=(784,)))
		aggregate_model.add(Dropout(0.2))
		aggregate_model.add(Dense(512, activation='relu'))
		aggregate_model.add(Dropout(0.2))
		aggregate_model.add(Dense(self.num_classes, activation='softmax'))

		avg_weights = sum([np.array(self.segment_models[segment].get_weights()) for segment in self.segment_models])/self.num_segments
		aggregate_model.set_weights(avg_weights)

		aggregate_model.compile(loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'])

		score = aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
		print(score)

		for segment in self.segment_models:
			self.segment_models[segment] = clone_model(aggregate_model)

		for segment in self.segment_models:
			# model_seg = self.segment_models[segment]
			(x_train_seg, y_train_seg) = self.segment_batches[segment]				
			# self.segment_models[segment] = self.train_model_segment(model_seg, x_train_seg, y_train_seg)
			self.segment_models[segment].compile(loss='categorical_crossentropy',
	            optimizer=Adam(),
	            metrics=['accuracy'])

			history = self.segment_models[segment].fit(x_train_seg, y_train_seg,
		        batch_size=self.batch_size,
		        epochs=1,
		        verbose=1,
		        validation_data=(self.x_test, self.y_test))

		avg_weights = sum([np.array(self.segment_models[segment].get_weights()) for segment in self.segment_models])/self.num_segments
		aggregate_model.set_weights(avg_weights)

		aggregate_model.compile(loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'])

		score = aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
		print(score)

		# count = 0
		# for segment in self.segment_models:
		# 	model_seg = self.segment_models[segment]
		# 	seg_weights = model_seg.get_weights()
		# 	print(seg_weights.shape)
		# 	if not weights:
		# 		weights = seg_weights
		# 		count += 1
		# 	else:
		# 		weights = (weights*count + seg_weights)
		# 		count += 1
		# 		weights /= count


	def train_model_segment(self, model, x_train, y_train):
		return model


model_instance = distributed_model_training()
# model_instance.get_data()
