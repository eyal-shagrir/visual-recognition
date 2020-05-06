from .classifier import Classifier
from data_utils import normalize_data
import numpy as np

CLASSES_NUM = 10
IMAGE_SIZE = 3072
DELTA = 1

def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

class SVM(Classifier):

    def __init__(self):
        super().__init__()
        self.training_set = None
        self.training_labels = None

    def train(self, training_set, training_labels, weights=None, bias=None, reg=0.0001):
        self.training_set = training_set
        self.training_labels = training_labels
        n = training_set.shape[0]
        if not weights:
            weights = np.random.rand(CLASSES_NUM, IMAGE_SIZE)
            # weights = np.full((10, 3072), 0.1)
        if not bias:
            bias = np.ones((CLASSES_NUM, 1))

        W = np.hstack((weights, bias))
        X = np.hstack((training_set, np.ones((n, 1))))
        y = training_labels

        iteration = 0
        while True:
            iteration += 1
            scores = W @ X.T
            scores_of_right_labels = scores[y, np.arange(n)]
            margins = np.maximum(scores - scores_of_right_labels + DELTA, 0)
            margins[y, np.arange(n)] = 0
            losses = np.sum(margins, axis=0)
            general_loss = np.average(losses) + reg * np.linalg.norm(W)

            if general_loss <= 100:
                print(f'i={iteration}, loss = {general_loss}')
                break

            indicators = margins
            losses_indices = np.nonzero(margins)
            indicators[losses_indices] = 1
            losses_sum = np.sum(indicators, axis=0)
            indicators[y, np.arange(n)] = -losses_sum
            gradient = (indicators @ X) / n

            W += 0.0001 * -gradient



        # new = W[:, :-1]
        return W


    def predict(self, testing_set, W):
        X = np.hstack((testing_set, np.ones((testing_set.shape[0], 1))))
        scores = W @ X.T
        y_pred = np.argmax(scores, axis=0)
        return y_pred