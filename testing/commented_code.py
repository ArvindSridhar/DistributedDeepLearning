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