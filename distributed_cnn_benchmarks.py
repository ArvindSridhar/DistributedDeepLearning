from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg as la
import random
import itertools
import operator
import time

class distributed_cnn_benchmark:

	def __init__(self):
		self.num_classes = 10
		self.batch_size = 500
		self.num_segments = 1
		self.num_training_iterations = 1
		self.num_iters_on_segment = 3
		self.cached_predictions = {}
		self.get_data()
		self.define_segment_models()
		self.train_model_aggregate()
		self.eval_model_aggregate()

	def tensorflow_device_test(self):
		hello = tf.constant('Tensorflow check passed')
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		print(sess.run(hello))

	def get_data(self):
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		self.cifar = True
		self.num_training_examples, self.num_test_examples = x_train.shape[0], x_test.shape[0]
		if self.cifar:
			self.img_rows, self.img_cols, self.num_channels = x_train.shape[2], x_train.shape[3], x_train.shape[1]
			x_train = x_train.reshape(self.num_training_examples, self.num_channels, self.img_rows, self.img_cols)
			x_test = x_test.reshape(self.num_test_examples, self.num_channels, self.img_rows, self.img_cols)
			self.input_shape = (self.num_channels, self.img_rows, self.img_cols)
		else:
			self.img_rows, self.img_cols, self.num_channels = x_train.shape[1], x_train.shape[2], 1
			x_train = x_train.reshape(self.num_training_examples, self.img_rows, self.img_cols, self.num_channels)
			x_test = x_test.reshape(self.num_test_examples, self.img_rows, self.img_cols, self.num_channels)
			self.input_shape = (self.img_rows, self.img_cols, self.num_channels)
		self.y_train = y_train.reshape(self.num_training_examples, 1)
		self.y_test = y_test.reshape(self.num_test_examples, 1)
		self.x_train = x_train.astype('float32')
		self.x_test = x_test.astype('float32')
		self.x_train /= 255
		self.x_test /= 255
		# Convert the y vectors to categorical format for crossentropy prediction
		self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
		self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

	def distribute_data(self):
		# Define how much data per segment
		self.segment_batches = {}
		data_per_segment = int(math.floor(self.num_training_examples/self.num_segments))

		# Shuffle x_train and y_train together
		rng_state = np.random.get_state()
		np.random.shuffle(self.x_train)
		np.random.set_state(rng_state)
		np.random.shuffle(self.y_train)

		# Distribute the shuffled data
		for i in range(self.num_segments):
			self.segment_batches["seg"+str(i)] = (self.x_train[data_per_segment*i:data_per_segment*i+data_per_segment],
												  self.y_train[data_per_segment*i:data_per_segment*i+data_per_segment])

	def get_new_model(self):
		# This is the model that the user defines/provides to us
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
			activation='relu',
			input_shape=self.input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(self.num_classes, activation='softmax'))
		return model

	def define_segment_models(self):
		self.segment_models = {}
		model = self.get_new_model()
		for i in range(self.num_segments):
			# Initialize each segment model using the same randomly-selected initial weights
			self.segment_models["seg"+str(i)] = clone_model(model)

	def train_segment(self, segment):
		segment = "seg"+str(segment)
		print('Segment:', segment)
		(x_train_seg, y_train_seg) = self.segment_batches[segment]
		model_seg = self.segment_models[segment]
		model_seg.compile(loss='categorical_crossentropy',
			optimizer=Adam(),
			metrics=['accuracy'])
		history = model_seg.fit(x_train_seg, y_train_seg,
			batch_size=self.batch_size,
			epochs=self.num_iters_on_segment,
			verbose=1,
			validation_data=(self.x_test, self.y_test))

	def train_model_aggregate(self):
		# Train the segment models, ideally in parallel
		start_time = time.time()
		for i in range(self.num_training_iterations):
			self.distribute_data()
			for segnum in list(range(self.num_segments)):
				self.train_segment(segnum)
		print('\n-------------------------------------------------------------------------------------------------')
		time_seg_training = time.time() - start_time
		print("Segment model (serial) training time:", time_seg_training, "seconds")

		# Ensemble the models and perform convolutional boosting to produce final predictions
		start_time2 = time.time()
		self.convolutional_boosted_ensemble_train()
		time_conv_training = time.time() - start_time2
		print("Convolutional ensemble training time:", time_conv_training, "seconds")
		total_train_time = time.time() - start_time
		print("Full training time:", total_train_time, "seconds")

		# Amdahl's Law calculation of potential training time
		s, P = time_conv_training/total_train_time, self.num_segments
		print("Potential training time (with parallelism):", total_train_time * (s + (1 - s)/P), "seconds")

	def eval_model_aggregate(self):
		# Evaluate the final model aggregate with training and test data
		print('-------------------------------------------------------------------------------------------------')
		start_time = time.time()
		train_score = self.convolutional_boosted_ensemble_evaluate(self.x_train, self.y_train)
		test_score = self.convolutional_boosted_ensemble_evaluate(self.x_test, self.y_test)
		print("Training set prediction accuracy of convolutional boosted ensemble:", train_score)
		print("Test set prediction accuracy of convolutional boosted ensemble:", test_score)
		print("Full evaluation time:", time.time() - start_time, "seconds")
		print('-------------------------------------------------------------------------------------------------\n')

	def predict_cached(self, model, segment, x_input):
		cached_key = segment + str(x_input.tobytes())
		if cached_key not in self.cached_predictions:
			self.cached_predictions[cached_key] = model.predict(x_input)
			# print("Didn't get from cache, had to recompute:", segment, x_input.shape)
		return self.cached_predictions[cached_key]

	def get_ensemble_predictions(self, x_input, expand_array=False):
		"""
			Gives you the classification predictions for some x_input from each trained segment model
			@param: x_input of the shape (num_examples, num_features)
			@return: an output of the shape (num_segments, num_examples) if expand_array is False
			@return: an output of the shape (num_examples, num_segments, num_classes, num_channels) if expand_array is True
		"""
		ensemble_predictions = []
		for segment in sorted(self.segment_models):
			model = self.segment_models[segment]
			if not expand_array:
				prediction = list(np.argmax(self.predict_cached(model, segment, x_input), axis=1))
			else:
				prediction = self.predict_cached(model, segment, x_input).T
			ensemble_predictions.append(prediction)
		if not expand_array:
			return np.array(ensemble_predictions)
		return np.array(ensemble_predictions).T.reshape((x_input.shape[0], self.num_segments, self.num_classes, 1))

	def convolutional_boosted_ensemble_train(self):
		"""
			Approach: you use part of the test set to gauge the veracity of each model, intelligently
			performed using a deep convolutional neural network to learn on its own how trustworthy
			each model is. Each model generates its own prediction for some input image, and these
			predictions are run through the convolutional ensemble model & a final prediction is given.
		"""
		# Break up the test set into the train_ensemble and test_ensemble sets
		split_value = self.num_test_examples//2
		x_train_ensemble, y_train_ensemble = self.x_test[0:split_value], self.y_test[0:split_value]
		x_test_ensemble, y_test_ensemble = self.x_test[split_value:], self.y_test[split_value:]

		# Define the convolutional ensemble model as a deep convolutional neural network
		self.conv_ensemble_model = Sequential()
		self.conv_ensemble_model.add(Conv2D(32, kernel_size=(3, 3),
			activation='relu',
			input_shape=(self.num_segments, self.num_classes, 1,)))
		self.conv_ensemble_model.add(Conv2D(64, (3, 3), activation='relu'))
		# self.conv_ensemble_model.add(MaxPooling2D(pool_size=(2, 2)))
		self.conv_ensemble_model.add(Dropout(0.25))
		self.conv_ensemble_model.add(Flatten())
		self.conv_ensemble_model.add(Dense(128, activation='relu'))
		self.conv_ensemble_model.add(Dropout(0.2))
		self.conv_ensemble_model.add(Dense(self.num_classes, activation='softmax'))

		# Compile the convolutional ensemble model
		self.conv_ensemble_model.compile(loss='categorical_crossentropy',
			optimizer=Adam(),
			metrics=['accuracy'])

		# Train the convolutional ensemble model with the train_ensemble data
		ensemble_predictions = self.get_ensemble_predictions(x_train_ensemble, True)
		history = self.conv_ensemble_model.fit(ensemble_predictions, y_train_ensemble,
			batch_size=self.batch_size,
			epochs=60,
			verbose=0,
			callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=0)])

		# Compute the accuracy of the convolutional ensemble model with the train_ensemble data
		training_score = self.convolutional_boosted_ensemble_evaluate(x_train_ensemble, y_train_ensemble)
		print("Convolutional ensemble model accuracy on ensemble training data:", training_score)

		# Validate the convolutional ensemble model with the test_ensemble data
		validation_score = self.convolutional_boosted_ensemble_evaluate(x_test_ensemble, y_test_ensemble)
		print("Convolutional ensemble model accuracy on ensemble test data:", validation_score)

	def convolutional_boosted_ensemble_evaluate(self, x_input, y_output):
		ensemble_predictions = self.get_ensemble_predictions(x_input, True)
		return self.conv_ensemble_model.evaluate(ensemble_predictions, y_output, verbose=0)[1]


class serial_cnn_benchmark:

	def __init__(self):
		self.num_classes = 10
		self.batch_size = 500
		self.epochs = 3
		self.get_data()
		self.define_model()
		self.train_model()
		self.print_eval_results()

	def tensorflow_device_test(self):
		hello = tf.constant('Tensorflow check passed')
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		print(sess.run(hello))

	def get_data(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		self.cifar = False
		self.num_training_examples, self.num_test_examples = x_train.shape[0], x_test.shape[0]
		if self.cifar:
			self.img_rows, self.img_cols, self.num_channels = x_train.shape[2], x_train.shape[3], x_train.shape[1]
			x_train = x_train.reshape(self.num_training_examples, self.num_channels, self.img_rows, self.img_cols)
			x_test = x_test.reshape(self.num_test_examples, self.num_channels, self.img_rows, self.img_cols)
			self.input_shape = (self.num_channels, self.img_rows, self.img_cols)
		else:
			self.img_rows, self.img_cols, self.num_channels = x_train.shape[1], x_train.shape[2], 1
			x_train = x_train.reshape(self.num_training_examples, self.img_rows, self.img_cols, self.num_channels)
			x_test = x_test.reshape(self.num_test_examples, self.img_rows, self.img_cols, self.num_channels)
			self.input_shape = (self.img_rows, self.img_cols, self.num_channels)
		self.y_train = y_train.reshape(self.num_training_examples, 1)
		self.y_test = y_test.reshape(self.num_test_examples, 1)
		self.x_train = x_train.astype('float32')
		self.x_test = x_test.astype('float32')
		self.x_train /= 255
		self.x_test /= 255
		# Convert the y vectors to categorical format for crossentropy prediction
		self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
		self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

	def define_model(self):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
			activation='relu',
			input_shape=self.input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(self.num_classes, activation='softmax'))
		self.model = model

	def train_model(self):
		start_time = time.time()
		self.model.compile(loss='categorical_crossentropy',
			optimizer=Adam(),
			metrics=['accuracy'])
		history = self.model.fit(self.x_train, self.y_train,
			batch_size=self.batch_size,
			epochs=self.epochs,
			verbose=1,
			validation_data=(self.x_test, self.y_test))
		print('\n-------------------------------------------------------------------------------------------------')
		print("Serial CNN training time:", time.time() - start_time, "seconds")

	def print_eval_results(self):
		print('-------------------------------------------------------------------------------------------------')
		start_time = time.time()
		train_score = self.evaluate_model(self.x_train, self.y_train)
		test_score = self.evaluate_model(self.x_test, self.y_test)
		print("Training set prediction accuracy of serial CNN:", train_score)
		print("Test set prediction accuracy of serial CNN:", test_score)
		print("Full evaluation time:", time.time() - start_time, "seconds")
		print('-------------------------------------------------------------------------------------------------')

	def evaluate_model(self, x_input, y_output):
		return self.model.evaluate(x_input, y_output, verbose=0)[1]


distributed_cnn_benchmark_inst = distributed_cnn_benchmark()
serial_cnn_benchmark_inst = serial_cnn_benchmark()
