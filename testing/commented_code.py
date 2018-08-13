		#--------------------------------------------------------------------------------

		# for segment in self.segment_models:
		# 	score = self.segment_models[segment].evaluate(self.x_test, self.y_test, verbose=0)
		# 	print('[Segment ' + str(segment) + '] Test loss: ' + str(score[0]))
		# 	print('[Segment ' + str(segment) + '] Test accuracy: ' + str(score[1]))

		#--------------------------------------------------------------------------------

		# aggregate_model = Sequential()
		# aggregate_model.add(Dense(512, activation='relu', input_shape=(784,)))
		# aggregate_model.add(Dropout(0.2))
		# aggregate_model.add(Dense(512, activation='relu'))
		# aggregate_model.add(Dropout(0.2))
		# aggregate_model.add(Dense(self.num_classes, activation='softmax'))

		# avg_weights = sum([np.array(self.segment_models[segment].get_weights()) for segment in self.segment_models])/self.num_segments
		# aggregate_model.set_weights(avg_weights)

		# aggregate_model.compile(loss='categorical_crossentropy',
  #           optimizer=Adam(),
  #           metrics=['accuracy'])

		# score = aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
		# print(score)

		# for segment in self.segment_models:
		# 	self.segment_models[segment] = clone_model(aggregate_model)

		# for segment in self.segment_models:
		# 	# model_seg = self.segment_models[segment]
		# 	(x_train_seg, y_train_seg) = self.segment_batches[segment]				
		# 	# self.segment_models[segment] = self.train_model_segment(model_seg, x_train_seg, y_train_seg)
		# 	self.segment_models[segment].compile(loss='categorical_crossentropy',
	 #            optimizer=Adam(),
	 #            metrics=['accuracy'])

		# 	history = self.segment_models[segment].fit(x_train_seg, y_train_seg,
		#         batch_size=self.batch_size,
		#         epochs=5,
		#         verbose=1,
		#         validation_data=(self.x_test, self.y_test))

		# avg_weights = sum([np.array(self.segment_models[segment].get_weights()) for segment in self.segment_models])/self.num_segments
		# aggregate_model.set_weights(avg_weights)

		# aggregate_model.compile(loss='categorical_crossentropy',
  #           optimizer=Adam(),
  #           metrics=['accuracy'])

		# score = aggregate_model.evaluate(self.x_test, self.y_test, verbose=1)
		# print(score)

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

		# def train_model_segment(self, model, x_train, y_train):
		# return model







#colors = iter(['red', 'blue', 'green', 'black', 'yellow', 'teal', 'magenta', 'pink', 'skyblue', 'cyan'])






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













		# train_prediction_merged = np.argmax(self.aggregate_model.predict(self.x_test), axis=1)
		# print(train_prediction_merged)
		# print(np.argmax(self.y_test, axis=1))
		# diff_predictions = train_prediction_merged - np.argmax(self.y_test, axis=1)
		# misclassifications = np.count_nonzero(diff_predictions)
		# print((self.x_test.shape[0] - misclassifications)/float(self.x_test.shape[0]))
		# return




























# 	def get_ensemble_predictions(self, x_input):
# 		"""
# 			Gives you the classification predictions for some x_input from each trained segment model
# 			@param: x_input of the shape (num_test_examples, num_features)
# 			@return: an output of the shape (num_segments, num_test_examples). Predict output always squashed!
# 		"""
# 		ensemble_predictions = []
# 		for segment in sorted(self.segment_models):
# 			model = self.segment_models[segment]
# 			prediction = list(np.argmax(model.predict(x_input), axis=1))
# 			# prediction = keras.utils.to_categorical(np.argmax(model.predict(x_input), axis=1), self.num_classes)
# 			ensemble_predictions.append(prediction)
# 		return np.array(ensemble_predictions)

# 	def consensus_predict_ensemble_evaluate(self, x_input, y_output):
# 		y_output_labels = np.argmax(y_output, axis=1)
# 		ensemble_predictions = self.get_ensemble_predictions(x_input)
# 		consensus_predictions = np.zeros((x_input.shape[0]))
# 		for i in range(ensemble_predictions.shape[1]):
# 			column = list(ensemble_predictions[:, i])
# 			consensus_predictions[i] = int(self.utils.mode(column).item())
# 		diff_predictions = consensus_predictions - y_output_labels
# 		misclassifications = np.count_nonzero(diff_predictions)
# 		return (x_input.shape[0] - misclassifications)/float(x_input.shape[0])

# 	def neural_boosted_ensemble_train(self):
# 		"""
# 			Approach: you use part of the test set to gauge the veracity of each model, intelligently
# 			of course using a neural network to learn on its own how trustworthy each model is. Each
# 			model generates its own prediction for some input image, and these predictions are then
# 			run through the neural ensemble model and a final prediction is given
# 		"""
# 		# Break up the test set into the train_ensemble and test_ensemble sets
# 		test_set_size = 7000#self.x_test.shape[0]//2
# 		x_train_ensemble, y_train_ensemble = self.x_test[0:test_set_size], self.y_test[0:test_set_size]
# 		x_test_ensemble, y_test_ensemble = self.x_test[test_set_size:], self.y_test[test_set_size:]

# 		# Define the neural ensemble model as a simple deep neural network
# 		self.neural_ensemble_model = Sequential()
# 		self.neural_ensemble_model.add(Dense(128, activation='relu', input_shape=(self.num_segments,)))
# 		self.neural_ensemble_model.add(Dropout(0.2))
# 		self.neural_ensemble_model.add(Dense(128, activation='relu'))
# 		self.neural_ensemble_model.add(Dropout(0.2))
# 		self.neural_ensemble_model.add(Dense(self.num_classes, activation='softmax'))

# 		# Compile the neural ensemble model
# 		self.neural_ensemble_model.compile(loss='categorical_crossentropy',
# 			optimizer=Adam(),
# 			metrics=['accuracy'])

# 		# Train the neural ensemble model with the train_ensemble data
# 		ensemble_predictions = self.get_ensemble_predictions(x_train_ensemble).T
# 		history = self.neural_ensemble_model.fit(ensemble_predictions, y_train_ensemble,
# 	        batch_size=self.batch_size,
# 	        epochs=60,
# 	        verbose=1)

# 		# Validate the neural ensemble model with the test_ensemble data
# 		validation_score = self.neural_boosted_ensemble_evaluate(x_test_ensemble, y_test_ensemble)
# 		print("Neural ensemble model accuracy on ensemble test data:", validation_score)
# 		test_ensemble_predictions = self.get_ensemble_predictions(x_test_ensemble).T
# 		validation_score = self.neural_ensemble_model.evaluate(test_ensemble_predictions, y_test_ensemble, verbose=0)
# 		print("Neural ensemble model accuracy on ensemble test data:", validation_score)

# 		#testing, to_delete
# 		test_ensemble_predictions = self.get_ensemble_predictions(x_train_ensemble).T
# 		validation_score = self.neural_ensemble_model.evaluate(test_ensemble_predictions, y_train_ensemble, verbose=0)
# 		print(validation_score)
# 		test_ensemble_predictions = self.get_ensemble_predictions(x_train_ensemble).T
# 		validation_score = self.neural_ensemble_model.evaluate(test_ensemble_predictions, y_train_ensemble, verbose=0)
# 		print(validation_score)

# 		#Think in terms of a single val example: input this 1 by 784 row vector (one image),
# 		#each model outputs a prediction, predictions multiplied by weights, softmaxed at the end to get final prediction

# 		#Convolutional layer run over the 2D array output that is ensemble_predictions? Maybe run convolution layer over
# 		#the actual segment models themselves, over their weight vectors, to get compression?
# 		#-Nope, because this is really just a collection of training examples, can't really do this

# 		#You can run the training, val, and test data on this, but want to default to using the test data (last 5000 examples)

# 		#HYPERPARAM TUNING: increase # segments, decrease # layers, increase # epochs, dropout, size of each layer

# 		# ISSUES
# 		# 1) Too little data being trained on vs tested on (but that is also a prob with each indiv. segment)
# 		# 2) Too complex of a model, have overfitting issues, too many epochs
# 		# 3) Too long to do: instead, intelligently choose only a subset of the NNs to use for each predict, preferrably use a subset of NNs that are predicted to produce the most accurate result for a given input

# 	def neural_boosted_ensemble_evaluate(self, x_input, y_output):
# 		y_output_labels = np.argmax(y_output, axis=1)
# 		ensemble_predictions = self.get_ensemble_predictions(x_input).T
# 		#predict and one-hot encode instead
# 		model_predictions = np.argmax(self.neural_ensemble_model.predict(ensemble_predictions), axis=1)
# 		diff_predictions = model_predictions - y_output_labels
# 		misclassifications = np.count_nonzero(diff_predictions)
# 		return (x_input.shape[0] - misclassifications)/float(x_input.shape[0])
# 		#return self.neural_ensemble_model.evaluate(ensemble_predictions, y_output, verbose=0)[1]