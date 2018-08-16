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
import pandas
import matplotlib.pyplot as plt
import math
from scipy import linalg as la
import random
import itertools
import operator
import time

class distributed_cnn_training:

	def __init__(self):
		self.num_classes = 10
		self.num_grand_epochs = 1 #Can tune
		self.batch_size = 1000 #Can tune
		self.num_segments = 10 #Can tune
		self.num_iters_on_segment = 1 #Can tune
		self.cached_predictions = {}
		self.utils = utilities()
		self.get_data()
		self.distribute_data()
		self.define_segment_models()
		self.train_model_aggregate()

	def tensorflow_test(self):
		hello = tf.constant('Hello, TensorFlow!')
		sess = tf.Session()
		print(sess.run(hello))

	def get_data(self):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		self.img_rows, self.img_cols, self.num_channels = x_train.shape[1], x_train.shape[2], 1
		self.num_training_examples, self.num_test_examples = x_train.shape[0], x_test.shape[0]
		x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, self.num_channels)
		x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, self.num_channels)
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
		# Shuffle? np.random.shuffle(self.x_train)
		self.segment_batches = {}
		data_per_segment = int(math.floor(self.num_training_examples/self.num_segments))
		for i in range(self.num_segments):
			self.segment_batches["seg"+str(i)] = (self.x_train[data_per_segment*i:data_per_segment*i+data_per_segment],
												  self.y_train[data_per_segment*i:data_per_segment*i+data_per_segment])

	def get_new_model(self):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
			activation='relu',
			input_shape=(self.img_rows, self.img_cols, self.num_channels,)))
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
		self.segment_colors = {}
		model = self.get_new_model()
		for i in range(self.num_segments):
			# Initialize each segment model using the same randomly-selected initial weights
			self.segment_models["seg"+str(i)] = clone_model(model)
			self.segment_colors["seg"+str(i)] = self.utils.random_color()

	def train_model_aggregate(self):
		# Training and evaluation loop
		for i in range(self.num_grand_epochs):
			print("Grand Epoch:", i+1, "/", self.num_grand_epochs)
			
			# Re-define the aggregate model (stored on the master node, and ultimately returned), also re-initialize its weights
			self.aggregate_model = self.get_new_model()

			# Define a plotting object for every numpy array that comprises the weights of our neural network, only if the algorithm is on its last grand epoch
			if i == self.num_grand_epochs+1:
				self.plots = [pca_weights_plotter() for j in range(len(self.aggregate_model.get_weights()))]

			# Train individual models for specified number of epochs	
			for segment in sorted(self.segment_models):
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
				if i == self.num_grand_epochs+1:
					weights = model_seg.get_weights()
					for j in range(len(weights)):
						plot = self.plots[j]
						plot.plot_data(weights[j], self.segment_colors[segment])

			# Average the weights of the trained models on the segments, add these weights to the aggregate model
			avg_weights = sum([np.array(self.segment_models[segment].get_weights())*np.random.random()*32
							   for segment in self.segment_models])/self.num_segments
			self.aggregate_model.set_weights(avg_weights)

			# Compile aggregate model
			self.aggregate_model.compile(loss='categorical_crossentropy',
				optimizer=Adam(),
				metrics=['accuracy'])

			# # Evaluate aggregate model on the test set
			# score = self.aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
			# print("Aggregate model accuracy on test set:", score[1])

			# Plot the average model's weights and show the plots, only if the algorithm is on its last grand epoch
			if i == self.num_grand_epochs+1:
				avg_weights = self.aggregate_model.get_weights()
				for j in range(len(avg_weights)):
					plot = self.plots[j]
					plot.plot_data(avg_weights[j], "dark orange", 'x')
					plot.show_plot()

			# Redistribute the aggregate model to each segment for the next grand epoch of training, if not on last grand epoch
			if i != self.num_grand_epochs-1:
				for segment in self.segment_models:
					self.segment_models[segment] = clone_model(self.aggregate_model)

		print('')
		print('-------------------------------------------------------------------------------------------------')

		# Conduct final testing with the weight-average aggregate model approach (non-ensemble)
		train_score_merged = self.aggregate_model.evaluate(self.x_train, self.y_train, verbose=0)
		test_score_merged = self.aggregate_model.evaluate(self.x_test, self.y_test, verbose=0)
		print("Training set prediction accuracy with aggregate model:", train_score_merged[1])
		print("Test set prediction accuracy with aggregate model:", test_score_merged[1])

		print('-------------------------------------------------------------------------------------------------')

		# Conduct final testing with the consensus prediction ensemble approach, [include aggregate model in the ensemble]
		# self.segment_models['agg'] = self.aggregate_model
		train_score_consensus = self.consensus_predict_ensemble_evaluate(self.x_train, self.y_train)
		test_score_consensus = self.consensus_predict_ensemble_evaluate(self.x_test, self.y_test)
		print("Training set prediction accuracy with consensus prediction ensembling:", train_score_consensus)
		print("Test set prediction accuracy with consensus prediction ensembling:", test_score_consensus)

		print('-------------------------------------------------------------------------------------------------')

		# Conduct final testing with the neural boosted ensemble approach
		self.neural_boosted_ensemble_train()
		train_score_neural = self.neural_boosted_ensemble_evaluate(self.x_train, self.y_train)
		test_score_neural = self.neural_boosted_ensemble_evaluate(self.x_test, self.y_test)
		print("Training set prediction accuracy with neural boosted ensembling:", train_score_neural)
		print("Test set prediction accuracy with neural boosted ensembling:", test_score_neural)

		print('-------------------------------------------------------------------------------------------------')

		# Conduct final testing with the convolutional boosted ensemble approach
		# assert self.num_classes == self.num_segments, "Cannot perform convolutional ensembling at the moment"
		self.convolutional_boosted_ensemble_train()
		train_score_convolutional = self.convolutional_boosted_ensemble_evaluate(self.x_train, self.y_train)
		test_score_convolutional = self.convolutional_boosted_ensemble_evaluate(self.x_test, self.y_test)
		print("Training set prediction accuracy with convolutional boosted ensembling:", train_score_convolutional)
		print("Test set prediction accuracy with convolutional boosted ensembling:", test_score_convolutional)

		print('-------------------------------------------------------------------------------------------------')

	def predict_cached(self, model, x_input):
		if x_input.tobytes() not in self.cached_predictions:
			self.cached_predictions[x_input.tobytes()] = model.predict(x_input)
			# print("Didn't get from cache, had to recompute:", x_input.shape)
		return self.cached_predictions[x_input.tobytes()]

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
				prediction = list(np.argmax(self.predict_cached(model, x_input), axis=1))
			else:
				prediction = self.predict_cached(model, x_input).T
			ensemble_predictions.append(prediction)
		if not expand_array:
			return np.array(ensemble_predictions)
		return np.array(ensemble_predictions).T.reshape((x_input.shape[0], self.num_segments, self.num_classes, 1))

	def consensus_predict_ensemble_evaluate(self, x_input, y_output):
		y_output_labels = np.argmax(y_output, axis=1)
		ensemble_predictions = self.get_ensemble_predictions(x_input)
		consensus_predictions = np.zeros((x_input.shape[0]))
		for i in range(ensemble_predictions.shape[1]):
			column = list(ensemble_predictions[:, i])
			consensus_predictions[i] = int(self.utils.mode(column).item())
		diff_predictions = consensus_predictions - y_output_labels
		misclassifications = np.count_nonzero(diff_predictions)
		return (x_input.shape[0] - misclassifications)/float(x_input.shape[0])

	def neural_boosted_ensemble_train(self):
		"""
			Approach: you use part of the test set to gauge the veracity of each model, intelligently
			performed using a neural network to learn on its own how trustworthy each model is. Each
			model generates its own prediction for some input image, and these predictions are then
			run through the neural ensemble model and a final prediction is given.
		"""
		# Break up the test set into the train_ensemble and test_ensemble sets
		test_set_size = self.x_test.shape[0]//2
		x_train_ensemble, y_train_ensemble = self.x_test[0:test_set_size], self.y_test[0:test_set_size]
		x_test_ensemble, y_test_ensemble = self.x_test[test_set_size:], self.y_test[test_set_size:]

		# Define the neural ensemble model as a simple deep neural network
		self.neural_ensemble_model = Sequential()
		self.neural_ensemble_model.add(Dense(512, activation='relu', input_shape=(self.num_segments,)))
		self.neural_ensemble_model.add(Dropout(0.3))
		self.neural_ensemble_model.add(Dense(512, activation='relu'))
		self.neural_ensemble_model.add(Dropout(0.3))
		self.neural_ensemble_model.add(Dense(512, activation='relu'))
		self.neural_ensemble_model.add(Dropout(0.3))
		self.neural_ensemble_model.add(Dense(self.num_classes, activation='softmax'))

		# Compile the neural ensemble model
		self.neural_ensemble_model.compile(loss='categorical_crossentropy',
			optimizer=Adam(),
			metrics=['accuracy'])

		# Train the neural ensemble model with the train_ensemble data
		ensemble_predictions = self.get_ensemble_predictions(x_train_ensemble).T
		history = self.neural_ensemble_model.fit(ensemble_predictions, y_train_ensemble,
			batch_size=self.batch_size,
			epochs=60,
			verbose=0,
			callbacks=[EarlyStopping(monitor='loss', patience=5, verbose=0)])

		# Compute the accuracy of the neural ensemble model with the train_ensemble data
		training_score = self.neural_boosted_ensemble_evaluate(x_train_ensemble, y_train_ensemble)
		print("Neural ensemble model accuracy on ensemble training data:", training_score)

		# Validate the neural ensemble model with the test_ensemble data
		validation_score = self.neural_boosted_ensemble_evaluate(x_test_ensemble, y_test_ensemble)
		print("Neural ensemble model accuracy on ensemble test data:", validation_score)

	def neural_boosted_ensemble_evaluate(self, x_input, y_output):
		ensemble_predictions = self.get_ensemble_predictions(x_input).T
		return self.neural_ensemble_model.evaluate(ensemble_predictions, y_output, verbose=0)[1]

	def convolutional_boosted_ensemble_train(self):
		"""
			Approach: you use part of the test set to gauge the veracity of each model, intelligently
			performed using a deep convolutional neural network to learn on its own how trustworthy
			each model is. Each model generates its own prediction for some input image, and these
			predictions are run through the convolutional ensemble model & a final prediction is given.
		"""
		# Break up the test set into the train_ensemble and test_ensemble sets
		test_set_size = self.x_test.shape[0]//2
		x_train_ensemble, y_train_ensemble = self.x_test[0:test_set_size], self.y_test[0:test_set_size]
		x_test_ensemble, y_test_ensemble = self.x_test[test_set_size:], self.y_test[test_set_size:]

		# Define the convolutional ensemble model as a deep convolutional neural network
		self.conv_ensemble_model = Sequential()
		self.conv_ensemble_model.add(Conv2D(32, kernel_size=(3, 3),
			activation='relu',
			input_shape=(self.num_segments, self.num_classes, 1,)))
		self.conv_ensemble_model.add(Conv2D(64, (3, 3), activation='relu'))
		self.conv_ensemble_model.add(MaxPooling2D(pool_size=(2, 2)))
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
			epochs=40,
			verbose=0,
			callbacks=[EarlyStopping(monitor='loss', patience=5, verbose=0)])

		# Compute the accuracy of the convolutional ensemble model with the train_ensemble data
		training_score = self.convolutional_boosted_ensemble_evaluate(x_train_ensemble, y_train_ensemble)
		print("Convolutional ensemble model accuracy on ensemble training data:", training_score)

		# Validate the convolutional ensemble model with the test_ensemble data
		validation_score = self.convolutional_boosted_ensemble_evaluate(x_test_ensemble, y_test_ensemble)
		print("Convolutional ensemble model accuracy on ensemble test data:", validation_score)

	def convolutional_boosted_ensemble_evaluate(self, x_input, y_output):
		ensemble_predictions = self.get_ensemble_predictions(x_input, True)
		return self.conv_ensemble_model.evaluate(ensemble_predictions, y_output, verbose=0)[1]


class utilities:

	def random_color(self):
		return "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])

	def mode(self, L):
		SL = sorted((x, i) for i, x in enumerate(L))
		groups = itertools.groupby(SL, key=operator.itemgetter(0))
		def auxfun(g):
			item, iterable = g
			count = 0
			min_index = len(L)
			for _, where in iterable:
				count += 1
				min_index = min(min_index, where)
			return count, -min_index
		return max(groups, key=auxfun)[0]


class pca_weights_plotter:

	def __init__(self):
		self.fig = plt.figure()
		
	def plot_data(self, data, color, symbol='.', num_dims_to_keep=2):
		self.data = data
		if len(data.shape) < 2:
			self.m, self.n = self.data.shape, 1
		else:
			self.m, self.n = self.data.shape
		self.num_dims_to_keep = num_dims_to_keep
		self.color = color
		self.symbol = symbol
		self.plot_PCA()
		# self.test_PCA()

	def PCA(self):
		data_mean_normalized = self.data - self.data.mean(axis=0)
		R = np.cov(data_mean_normalized, rowvar=False)
		evals, evecs = la.eigh(R)
		idx = np.argsort(evals)[::-1]
		evecs = evecs[:,idx]
		evals = evals[idx]
		evecs = evecs[:, :self.num_dims_to_keep]
		return np.dot(evecs.T, data_mean_normalized.T).T, evals, evecs

	def test_PCA(self):
		_, _, eigenvectors = self.PCA()
		data_recovered = np.dot(eigenvectors, self.m).T
		data_recovered += data_recovered.mean(axis=0)
		assert np.allclose(self.data, data_recovered)

	def plot_PCA(self):
		ax1 = self.fig.add_subplot(111)
		data_resc, evals, evecs = self.PCA()
		ax1.plot(data_resc[:, 0], data_resc[:, 1], self.symbol, mfc=self.color, mec=self.color)

	def show_plot(self):
		plt.show()


model_instance = distributed_cnn_training()
