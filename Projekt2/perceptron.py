import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions


class Perceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.errors = None
        self.w = None
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w = np.zeros(1 + x.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    # we classify only two classes, so we arbitrarily choose that
    # we will be classifying class=2 and other (1 and 0)
    y_train[y_train != 2] = -1
    y_train[y_train == 2] = 1

    perceptron = Perceptron(eta=0.1, n_iter=1000)
    perceptron.fit(x_train, y_train)

    # same trick for test data -> bc we need only to classes
    y_test[y_test != 2] = -1
    y_test[y_test == 2] = 1
    plot_decision_regions(x=x_test, y=y_test, classifier=perceptron)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
