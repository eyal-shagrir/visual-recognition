from .classifier import Classifier
import numpy as np


class KNN(Classifier):

    def __init__(self):
        super().__init__()
        self.training_set = None
        self.training_labels = None
        self.k = None
        self.num_loops = None

    def train(self, training_set, training_labels, k=1, num_loops=0):
        self.training_set = training_set
        self.training_labels = training_labels
        self.k = k
        self.num_loops = num_loops
        return {}

    def predict(self, testing_set):
        num_loops = self.num_loops
        if num_loops == 2:
            dists = self.compute_distances_two_loops(testing_set)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(testing_set)
        else:
            dists = self.compute_distances_no_loops(testing_set)

        return self.predict_labels(dists)

    def compute_distances_two_loops(self, testing_set):
        num_test = testing_set.shape[0]
        num_train = self.training_set.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.power(testing_set[i] - self.training_set[j], 2)))
        return dists

    def compute_distances_one_loop(self, testing_set):
        num_test = testing_set.shape[0]
        num_train = self.training_set.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum(np.power(self.training_set - testing_set[i, :], 2), axis=1))
        return dists

    def compute_distances_no_loops(self, testing_set):
        # calculations are based on the formula (x-y)^2 = x^2 -2xy + y^2
        tests_square_sums = np.sum(np.power(testing_set, 2), axis=1)
        training_square_sums = np.sum(np.power(self.training_set, 2), axis=1)
        tests_training_mul = testing_set @ self.training_set.T
        tests_training_mul *= -2
        dists = tests_training_mul
        dists += tests_square_sums.reshape(tests_square_sums.shape[0], 1)
        dists += training_square_sums.reshape(1, training_square_sums.shape[0])
        return dists

    def predict_labels(self, dists):
        k = self.k

        # sort distances and get indices in accordance
        sorted_dists_indices = np.argsort(dists, axis=1)

        # get matching labels
        labels = np.apply_along_axis(lambda a: self.training_labels[a], 1, sorted_dists_indices)

        # get most frequent label in the first k labels
        y_pred = np.apply_along_axis(lambda a: np.argmax(np.bincount(a[:k])), 1, labels)

        return y_pred
