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
from scipy import linalg as la
import random
import itertools
import operator

class distributed_model_training:

	def __init__(self):
		self.num_classes = 10
		self.num_grand_epochs = 2 #Can tune
		self.batch_size = 100 #Can tune
		self.num_segments = 10 #Can tune
		self.num_iters_on_segment = 3 #Can tune
		self.utils = utilities()
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
		self.segment_colors = {}
		for i in range(self.num_segments):
			model = Sequential()
			model.add(Dense(512, activation='relu', input_shape=(784,)))
			model.add(Dropout(0.2))
			model.add(Dense(512, activation='relu'))
			model.add(Dropout(0.2))
			model.add(Dense(self.num_classes, activation='softmax'))
			self.segment_models["seg"+str(i)] = model
			self.segment_colors["seg"+str(i)] = self.utils.random_color()

	def train_model_aggregate(self):
		# Training and evaluation loop
		for i in range(self.num_grand_epochs):
			print("Grand Epoch:", i+1, "/", self.num_grand_epochs)
			
			# Re-define the aggregate model (stored on the master node, and ultimately returned), also re-initialize its weights
			self.aggregate_model = Sequential()
			self.aggregate_model.add(Dense(512, activation='relu', input_shape=(784,)))
			self.aggregate_model.add(Dropout(0.2))
			self.aggregate_model.add(Dense(512, activation='relu'))
			self.aggregate_model.add(Dropout(0.2))
			self.aggregate_model.add(Dense(self.num_classes, activation='softmax'))

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
			avg_weights = sum([np.array(self.segment_models[segment].get_weights())*np.random.random()*32 for segment in self.segment_models])/self.num_segments
			self.aggregate_model.set_weights(avg_weights)

			# Compile aggregate model
			self.aggregate_model.compile(loss='categorical_crossentropy',
				optimizer=Adam(),
				metrics=['accuracy'])

			# Evaluate aggregate model on the test set
			score = self.aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
			print(score)

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

		# Conduct final testing of aggregate model
		train_score_merged = self.aggregate_model.evaluate(self.x_train, self.y_train, verbose=1)
		test_score_merged = self.aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
		print("Training accuracy with merged model:", train_score_merged[1])
		print("Test accuracy with merged model:", test_score_merged[1])

		# Consensus prediction approach (ensembling neural network models)
		# score = 0
		# for i in range(len(self.x_test)):
		# 	x_test_example = self.x_test[i].reshape(1, 784)
		# 	y_label_example = np.argmax(self.y_test[i])
		# 	ensemble_predictions = {}
		# 	for segment in self.segment_models:
		# 		model = self.segment_models[segment]
		# 		prediction = np.argmax(model.predict(x_test_example))
		# 		ensemble_predictions[prediction] = ensemble_predictions.get(prediction, 0) + 1
		# 	consensus_prediction = max(ensemble_predictions, key=ensemble_predictions.get)
		# 	if consensus_prediction == y_label_example:
		# 		score += 1
		# print("Test accuracy with ensembling and consensus predictors (non-vectorized):", score/10000.0)

		# Vectorized consensus prediction, include aggregate model in the ensemble
		self.segment_models['agg'] = self.aggregate_model
		train_score_consensus = self.consensus_predict(self.x_train, self.y_train)
		test_score_consensus = self.consensus_predict(self.x_test, self.y_test)
		print("Training accuracy with ensembling and consensus predictors:", train_score_consensus)
		print("Test accuracy with ensembling and consensus predictors:", test_score_consensus)

		# Maybe add aggregate model prediction to consensus as a tiebreaker
		# have each model initialized at the same starting point, and then have it run for a few iterations so that they aren't too divergent
		# output the results of the model to file
		# progressively increase the number of iterations with each grand epoch

		#Instead of blind consensus prediction, pick using intelligent strategies, like if model succeeded on this example before during training or something, obviously pick it (or a similar example, gauge similarity betw. examples), so a weighted consensus

	def consensus_predict(self, x_test, y_test):
		y_test_labels = np.argmax(y_test, axis=1)
		ensemble_predictions = []
		for segment in self.segment_models:
			model = self.segment_models[segment]
			prediction = list(np.argmax(model.predict(x_test), axis=1))
			ensemble_predictions.append(prediction)
		consensus_predictions = np.zeros((x_test.shape[0]))
		ensemble_predictions = np.array(ensemble_predictions)
		for i in range(ensemble_predictions.shape[1]):
			column = list(ensemble_predictions[:, i])
			consensus_predictions[i] = int(self.utils.mode(column).item())
		diff_predictions = consensus_predictions - y_test_labels
		misclassifications = np.count_nonzero(diff_predictions)
		return (x_test.shape[0] - misclassifications)/float(x_test.shape[0])


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


model_instance = distributed_model_training()
