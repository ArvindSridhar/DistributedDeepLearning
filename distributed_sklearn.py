#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import print_function

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

import matplotlib.pyplot as plt
import cPickle
import gzip


def fetch_mnist():
    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_test_split(
        np.concatenate((train_set[0], valid_set[0], test_set[0])) / 255.,
        np.concatenate((train_set[1], valid_set[1], test_set[1])),
        train_size=60000)


class distributed_nn_training:
    def __init__(self, n_segments):
        self.n_classes = 3
        self.n_iterations = 100
        self.batch_size = 500
        self.n_segments = n_segments
        self.n_epochs = 10
        self.get_data()
        self.distribute_data()
        self.aggregate_model = self.define_segment_models()
        self.train_model_aggregate()

    def get_data(self):
        # iris = datasets.load_iris()
        # self.x_train, self.y_train = iris.data, iris.target

        self.x_train, self.x_test, self.y_train, self.y_test = fetch_mnist()
        print('Digit distribution in whole dataset:', np.bincount(self.y_train.astype('int64')))

    def distribute_data(self):
        self.segment_batches = []
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train)
        data_per_segment = int(math.floor(self.x_train.shape[0] / self.n_segments))
        for i in range(self.n_segments):
            self.segment_batches.append((self.x_train[data_per_segment * i:data_per_segment * i + data_per_segment],
                                         self.y_train[data_per_segment * i:data_per_segment * i + data_per_segment]))
        print([(s[0].shape, s[1].shape) for s in self.segment_batches])

    def get_new_model(self):
        iris_model = MLPClassifier(
            hidden_layer_sizes=[3, ],
            # activation='relu',
            solver='sgd',
            learning_rate='constant',
            max_iter=self.n_epochs,
            learning_rate_init=0.05,
            momentum=0,
            batch_size=self.batch_size,
            random_state=0)

        mnist_model = MLPClassifier(
            hidden_layer_sizes=[10, ],
            # activation='relu',
            solver='adam',
            # learning_rate='invscaling',
            learning_rate='constant',
            max_iter=self.n_epochs,
            learning_rate_init=0.01,
            momentum=0.95,
            nesterovs_momentum=True,
            batch_size=self.batch_size,
            random_state=0,
            verbose=False,
            tol=1e-20,
            early_stopping=False)
        return mnist_model

    def define_segment_models(self):
        self.segment_models = []
        common_model = self.get_new_model()
        for i in range(self.n_segments):
            self.segment_models.append(clone(common_model))
        return common_model

    def aggregate_models(self):
        try:
            self.aggregate_model.coefs_ = list(sum([np.array(s.coefs_)
                                                    for s in self.segment_models]) / self.n_segments)
            self.aggregate_model.intercepts_ = list(sum([np.array(s.intercepts_)
                                                         for s in self.segment_models]) / self.n_segments)
        except Exception as e:
            print(str(e))
            print([np.array(s.coefs_) for s in self.segment_models])
            print([np.array(s.intercepts_) for s in self.segment_models])
            raise

        self.aggregate_model.classes_ = self.segment_models[0].classes_[:]
        self.aggregate_model._label_binarizer = self.segment_models[0]._label_binarizer
        self.aggregate_model.n_layers_ = self.segment_models[0].n_layers_
        self.aggregate_model.n_outputs_ = self.segment_models[0].n_outputs_
        self.aggregate_model.out_activation_ = self.segment_models[0].out_activation_

    def reset(self, model):
        model.n_iter_ = 0
        model.t_ = 0
        model.loss_curve_ = []
        model._no_improvement_count = 0
        model.best_loss_ = np.inf

    def clone_models(self):
        for i in range(self.n_segments):
            self.segment_models[i] = clone(self.aggregate_model)
            self.reset(self.segment_models[i])
            self.segment_models[i].coefs_ = copy.deepcopy(self.aggregate_model.coefs_)
            self.segment_models[i].intercepts_ = copy.deepcopy(self.aggregate_model.intercepts_)
            self.segment_models[i].classes_ = self.aggregate_model.classes_[:]
            self.segment_models[i].n_layers_ = self.aggregate_model.n_layers_
            self.segment_models[i].n_outputs_ = self.aggregate_model.n_outputs_
            self.segment_models[i]._label_binarizer = self.aggregate_model._label_binarizer
            self.segment_models[i].out_activation_ = self.aggregate_model.out_activation_

    def train_model_aggregate(self):
        self.agg_model_scores = []
        self.segment_models_scores = [np.zeros(self.n_iterations) for i in range(self.n_segments)]

        # Training and evaluation loop
        for i in range(self.n_iterations):
            print("Iteration:", i + 1, "/", self.n_iterations, end=' - ')
            for seg_index, model_seg in enumerate(self.segment_models):
                (x_train_seg, y_train_seg) = self.segment_batches[seg_index]
                if i == 0:
                    model_seg.fit(x_train_seg, y_train_seg)
                else:
                    model_seg.partial_fit(x_train_seg, y_train_seg)
                model_score = model_seg.score(self.x_train, self.y_train)
                self.segment_models_scores[seg_index][i] = model_score
                # print("\t Segment model {} score = {}".format(seg_index, model_score))

            self.aggregate_models()
            self.clone_models()
            agg_score = self.aggregate_model.score(self.x_train, self.y_train)
            self.agg_model_scores.append(agg_score)
            print("Aggregate model score = {}".format(agg_score))
        print('-------------------------------------------------------------------------------------------------')


def plot_model_scores():
    suptitle = 'MNIST data classification'
    for n_segments in range(12, 13, 2):   # range(start, stop, step)
        model = distributed_nn_training(n_segments)

        fig = plt.figure()
        fig.suptitle(suptitle, fontweight='bold')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('Number of segments = {}'.format(model.n_segments))
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('Accuracy')

        x_axis = range(model.n_iterations)
        for scores in model.segment_models_scores:
            plt.plot(x_axis, scores, 'k-', alpha=0.2)
        plt.plot(x_axis, model.agg_model_scores, 'b-')
        plt.savefig('iris_agg_scores_seg_{}.png'.format(model.n_segments))


import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plot_model_scores()
