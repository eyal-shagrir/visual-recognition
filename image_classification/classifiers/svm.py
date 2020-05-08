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


def compute_loss(X, y, W, reg):
    n = X.shape[0]
    scores = W @ X.T
    scores_of_right_labels = scores[y, np.arange(n)]
    margins = np.maximum(scores - scores_of_right_labels + DELTA, 0)
    margins[y, np.arange(n)] = 0
    losses = np.sum(margins, axis=0)
    general_loss = np.average(losses) + 0.5 * reg * np.linalg.norm(W)
    return margins, general_loss


def compute_loss_gradient(X, y, W, margins, reg):
    n = X.shape[0]
    indicators = margins
    losses_indices = np.nonzero(margins)
    indicators[losses_indices] = 1
    losses_sum = np.sum(indicators, axis=0)
    indicators[y, np.arange(n)] = -losses_sum
    gradient = (indicators @ X) / n
    gradient += reg * W
    return gradient


class SVM(Classifier):

    def __init__(self):
        super().__init__()
        self.training_set = None
        self.training_labels = None

    def get_batched_data(self, iteration, batch_num, batch_size):
        low_idx = iteration % batch_num
        high_idx = (iteration + 1) % batch_num
        if high_idx == 0:
            high_idx = batch_num
        low_bar = low_idx * batch_size
        high_bar = high_idx * batch_size
        images_batch = self.training_set[low_bar: high_bar, :]
        X = np.hstack((images_batch, np.ones((batch_size, 1))))
        y = self.training_labels[low_bar: high_bar]
        return X, y

    def train(self, training_set, training_labels, alpha=1e-5, reg=5000, iterations=200, batch_num=1):
        self.training_set = training_set
        self.training_labels = training_labels

        weights = np.zeros((CLASSES_NUM, IMAGE_SIZE))
        bias = np.ones((CLASSES_NUM, 1))

        W = np.hstack((weights, bias))

        images_num = training_set.shape[0]
        batch_size = int(images_num / batch_num)

        for i in range(iterations):
            X, y = self.get_batched_data(i, batch_num, batch_size)

            margins, general_loss = compute_loss(X, y, W, reg)
            gradient = compute_loss_gradient(X, y, W, margins, reg)

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
