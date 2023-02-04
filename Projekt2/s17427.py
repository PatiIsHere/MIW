import argparse

import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from perceptron import Perceptron
from plotka import plot_decision_regions


class PerceptronMulti:
    """
    Class with 3 binary perceptrons.
    """

    def __init__(self, perceptron1, perceptron2, perceptron3):
        self.per1 = perceptron1
        self.per2 = perceptron2
        self.per3 = perceptron3

    def predict(self, x):
        return np.where(self.per1.predict(x) == 1, 0, np.where(self.per2.predict(x) == 1, 1,
                                                               np.where(self.per3.predict(x) == 1, 2, 1)))


class LogisticRegressionGD:
    """
    Class from provided assignment materials.
    Class wasn't imported due to additional method - 'get_probability'
    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.cost = None
        self.w = None
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.eta * x.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost.append(cost)
        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def activation(self, z):  # probability based on net_input
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)

    def get_probability(self, y):
        return self.activation(self.net_input(y))


class LRGDMulti:
    """
    Class with 3 binary logisticRegression.
    """

    def __init__(self, logregGD1, logregGD2, logregGD3):
        self.lrgd1 = logregGD1
        self.lrgd2 = logregGD2
        self.lrgd3 = logregGD3

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, x):
        return np.where(self.lrgd1.predict(x) == 1, 0, np.where(self.lrgd2.predict(x) == 1, 1,
                                                                np.where(self.lrgd3.predict(x) == 1, 2, 1)))

    def print_probability(self, x):
        """
        Function prints activation probability for all logisticRegrestion atributes.
        :param x:
        :return: None
        """
        print(f'Class 0: {round(self.lrgd1.get_probability(x).max(), 6) * 100}%')
        print(f'Class 1: {round(self.lrgd2.get_probability(x).max(), 6) * 100}%')
        print(f'Class 2: {round(self.lrgd3.get_probability(x).max(), 6) * 100}%')


def parse_arguments():
    parser = argparse.ArgumentParser(description='This is a script to compare Perceptron and Logistic regression GD')
    return parser.parse_args()


def main():
    """
    Function load iris dataset, train Perceptrons and LogisticRegressionGDs to classify 3 different classess.
    Finally, function prints graph for each type for comparison with additional prediction data for LogisticRegressions.
    :return: None
    """
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4, stratify=y)

    # [~][~][~]Perceptron[~][~][~]

    # make copies from original train set
    y_train_01_subset = y_train.copy()
    y_train_02_subset = y_train.copy()
    y_train_03_subset = y_train.copy()

    # modify arrays for perceptrons
    y_train_01_subset[y_train_01_subset != 0] = -1
    y_train_01_subset[y_train_01_subset == 0] = 1

    y_train_02_subset[y_train_02_subset != 1] = -1
    y_train_02_subset[y_train_02_subset == 1] = 1

    y_train_03_subset[y_train_03_subset != 2] = -1
    y_train_03_subset[y_train_03_subset == 2] = 1

    # create perceptrons
    per1 = Perceptron(eta=0.05, n_iter=1000)
    per2 = Perceptron(eta=0.05, n_iter=1000)
    per3 = Perceptron(eta=0.05, n_iter=1000)

    # train perceptrons
    per1.fit(x_train, y_train_01_subset)
    per2.fit(x_train, y_train_02_subset)
    per3.fit(x_train, y_train_03_subset)

    # create multiperceptron class with trained binary perceptrons
    classifierPer = PerceptronMulti(per1, per2, per3)

    # [~][~][~]LogisticRegression[~][~][~]

    # make copies from original train set
    y_train_04_subset = y_train.copy()
    y_train_05_subset = y_train.copy()
    y_train_06_subset = y_train.copy()

    # modify arrays for logisticRegressions
    y_train_04_subset[y_train_04_subset != 0] = -1
    y_train_04_subset[y_train_04_subset == 0] = 1
    y_train_04_subset[y_train_04_subset == -1] = 0

    y_train_05_subset[y_train_05_subset != 1] = -1
    y_train_05_subset[y_train_05_subset == 1] = 1
    y_train_05_subset[y_train_05_subset == -1] = 0

    y_train_06_subset[y_train_06_subset != 2] = -1
    y_train_06_subset[y_train_06_subset == 2] = 1
    y_train_06_subset[y_train_06_subset == -1] = 0

    # create logisticRegressions
    logregGD1 = LogisticRegressionGD(eta=0.1, n_iter=1000, random_state=3)
    logregGD2 = LogisticRegressionGD(eta=0.1, n_iter=1000, random_state=1)
    logregGD3 = LogisticRegressionGD(eta=0.1, n_iter=1000, random_state=1)

    # train logisticRegressions
    logregGD1.fit(x_train, y_train_04_subset)
    logregGD2.fit(x_train, y_train_05_subset)
    logregGD3.fit(x_train, y_train_06_subset)

    # create multiLogisticRegressions class with trained binary logisticRegressions
    classifierLRGD = LRGDMulti(logregGD1, logregGD2, logregGD3)

    # print probability for each logisticRegression in LRGDMulti
    for i in range(x_test.shape[0]):
        print(f'Sample: {x_test[i]}, Real class: {y_test[i]}')
        classifierLRGD.print_probability(x_test[i])

    # make graphs
    plt.subplot(1, 2, 1)
    plot_decision_regions(x=x_test, y=y_test, classifier=classifierPer)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.title("PERCEPTRON")

    plt.subplot(1, 2, 2)
    plot_decision_regions(x=x_test, y=y_test, classifier=classifierLRGD)
    plt.xlabel('Petal length')
    plt.legend(loc='upper left')
    plt.title('LRGD')
    plt.show()


if __name__ == '__main__':
    parse_arguments()
    main()
