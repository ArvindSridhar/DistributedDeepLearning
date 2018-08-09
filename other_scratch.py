from __future__ import print_function
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam


def tensorflow_test():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))


class test:
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
        print(self.x_train[0:2].shape)
        print(self.y_train.shape)
        print(self.x_train[59999])
        print(self.y_train[0:5])

t = test()
t.get_data()

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
		self.num_classes = 10
		self.total_num_epochs = 2
		self.batch_size = 100
		self.get_data()
		self.distribute_data()
		self.define_models()
# 		self.train_model_aggregate()

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
		for i in range(10):
			self.segment_batches["seg"+str(i)] = (self.x_train[6000*i:6000*i+6000], self.y_train[6000*i:6000*i+6000])

	def define_models(self):
		self.segment_models = {}
		for i in range(10):
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
			        verbose=0,
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

		aggregate_model.compile(loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy'])

		print(len(self.segment_models))

		a = [self.segment_models[segment].get_weights() for segment in self.segment_models]
		print(type(a[0]))

		avg_weights = sum([self.segment_models[segment].get_weights() for segment in self.segment_models])/len(self.segment_models)
		print(avg_weights.shape)

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

a = [np.array(model_instance.segment_models[segment].get_weights()) for segment in model_instance.segment_models]
print(type(a[0]))

print([type(a[i]) for i in range(10)])

sum(a)

print(sorted(model_instance.segment_models))

aggregate_model = Sequential()
aggregate_model.add(Dense(512, activation='relu', input_shape=(784,)))
aggregate_model.add(Dropout(0.2))
aggregate_model.add(Dense(512, activation='relu'))
aggregate_model.add(Dropout(0.2))
aggregate_model.add(Dense(10, activation='softmax'))
print([i.shape for i in aggregate_model.get_weights()])


