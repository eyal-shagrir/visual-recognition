from classifiers.knn import KNN
from classifiers.svm import SVM
from data_utils import get_data, subsample_data, normalize_data, show_image
from classification_utils import cross_validate_params, predict_success_rate, cross_validate_and_predict

CIFAR10_FOLDER = 'cifar10_data'

# KNN HYPERPARAMETERS
Ks = (1, 3, 5, 8, 10, 12, 15, 20, 50, 100)  # best is around 10
KNN_HP = {'k': Ks}

# SVM HYPERPARAMETERS
ALPHAS = (1e-11, 1e-7, 1e-3)  # best is around 1e-7
REGS = (1000, 5000, 10000, 25000, 50000)  # best is around 30000
SVM_HP = {'batch_num': (10,), 'iterations': (500,), 'alpha': ALPHAS, 'reg': REGS}

CLASSIFIERS_HP = {'KNN': KNN_HP, 'SVM': SVM_HP}


def cross_validate_all_classifiers(training_set, training_labels,
                                   testing_set, testing_labels):
    for classifier_name, classifier_cls in {'SVM': SVM, 'KNN': KNN}.items():
        classifier = classifier_cls()
        cross_validate_and_predict(classifier,
                                   training_set, training_labels,
                                   testing_set, testing_labels,
                                   **CLASSIFIERS_HP[classifier_name])


def main():
    training_images, training_labels, testing_images, testing_labels = get_data(CIFAR10_FOLDER)

    (training_set, training_labels,
     testing_set, testing_labels) = subsample_data(training_images, training_labels,
                                                   testing_images, testing_labels,
                                                   training_num=5000, testing_num=500)
    normalize_data(training_set, testing_set)
    cross_validate_all_classifiers(training_set, training_labels,
                                   testing_set, testing_labels)


if __name__ == '__main__':
    main()
