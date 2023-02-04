import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD:

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


def main():
    iris = datasets.load_iris()
    x = iris.data[:, [2, 3]]
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

    # we classify only two classes, so we arbitrarily choose that
    # we will be classifying class=2 and other (1 and 0)
    y_train[y_train != 2] = -1
    y_train[y_train == 2] = 1
    y_train[y_train == -1] = 0

    lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd.fit(x_train, y_train)

    # same trick for test data -> bc we need only to classes
    y_test[y_test != 2] = -1
    y_test[y_test == 2] = 1
    y_test[y_test == -1] = 0
    plot_decision_regions(x=x_test, y=y_test, classifier=lrgd)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
