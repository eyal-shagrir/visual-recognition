from .classifier import Classifier
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

    def train(self, training_set, training_labels, alpha=1e-3, reg=1e-5, iterations=100):
        self.training_set = training_set
        self.training_labels = training_labels

        n = training_set.shape[0]
        weights = np.zeros((CLASSES_NUM, IMAGE_SIZE))
        bias = np.ones((CLASSES_NUM, 1))

        W = np.hstack((weights, bias))
        X = np.hstack((training_set, np.ones((n, 1))))
        y = training_labels

        for iteration in range(iterations):
            scores = W @ X.T
            scores_of_right_labels = scores[y, np.arange(n)]
            margins = np.maximum(scores - scores_of_right_labels + DELTA, 0)
            margins[y, np.arange(n)] = 0
            losses = np.sum(margins, axis=0)
            general_loss = np.average(losses) + 0.5 * reg * np.linalg.norm(W)

            indicators = margins
            losses_indices = np.nonzero(margins)
            indicators[losses_indices] = 1
            losses_sum = np.sum(indicators, axis=0)
            indicators[y, np.arange(n)] = -losses_sum
            gradient = (indicators @ X) / n
            gradient += reg * W

            W -= alpha * gradient

        return {'weights': W}

    def predict(self, testing_set, weights=None):
        if weights is None:
            bias = np.ones((CLASSES_NUM, 1))
            W = np.hstack((weights, bias))
        else:
            W = weights
        X = np.hstack((testing_set, np.ones((testing_set.shape[0], 1))))
        scores = W @ X.T
        y_pred = np.argmax(scores, axis=0)
        return y_pred
