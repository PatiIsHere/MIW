import argparse
import random as random
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from termcolor import colored


def main(args):
    # get 4 random files to show on graph
    rnd = random.sample(range(1, 16), 4)

    # define size of plots
    plt.subplots(2, 2, figsize=(16, 12))

    currentData = []

    plots_to_show = 1;
    fileNum = 0

    for filename in os.listdir(args.mandatory_argument):
        fileNum += 1

        f = os.path.join(args.mandatory_argument, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(colored(f.title(), 'orange'))
            currentData = np.loadtxt(fname=f)
            currentData = np.round(currentData, 3)

        x = currentData[:, [0]]
        y = currentData[:, [1]]

        # make train & test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

        # train size
        testSize = len(x_train)

        # linear coefficients
        linear_w1 = ((testSize * np.sum(x_train * y_train)) - (np.sum(x_train) * np.sum(y_train))) / (
                    (testSize * np.sum(x_train ** 2)) - (np.sum(x_train) ** 2))
        linear_w2 = ((np.sum(y_train) * np.sum(x_train ** 2)) - (np.sum(x_train) * np.sum(x_train * y_train))) / (
                    (testSize * np.sum(x_train ** 2)) - (np.sum(x_train) ** 2))

        # linear pred
        y_predict_linear = (linear_w1 * x_test) + linear_w2

        # LSE linear
        LSE_linear = np.sum((y_test - y_predict_linear) ** 2)

        # quadratic matrix

        matrix_3x3 = np.linalg.inv(
            [[np.sum(x_train ** 4), np.sum(x_train ** 3), np.sum(x_train ** 2)],
             [np.sum(x_train ** 3), np.sum(x_train ** 2), np.sum(x_train)],
             [np.sum(x_train ** 2), np.sum(x_train), testSize]]
        )

        matrix_3x1 = [[np.sum((x_train ** 2) * y_train)],
                      [np.sum(x_train * y_train)],
                      [np.sum(y_train)]]

        # quadratic coefficients
        quadratic_w = np.dot(matrix_3x3, matrix_3x1)

        # quadratic pred
        y_predict_quadratic = (quadratic_w[0] * x_test ** 2) + (quadratic_w[1] * x_test) + quadratic_w[2]

        # LSE quadratic
        LSE_quadratic = np.sum((y_test - y_predict_quadratic) ** 2)

        # Print data
        print(f'Linear Coefficient: w1 = {linear_w1}, w0 = {linear_w2}')
        print(f'Quadratic Coefficient: w2: = {quadratic_w[0]}, w1 = {quadratic_w[1]}, w0 = {quadratic_w[2]}')
        print(f'Linear LSE: {LSE_linear}')
        print(f'Quadratic LSE: {LSE_quadratic}')

        # choose what is better
        if LSE_linear < LSE_quadratic:
            print(colored('Linear model is better', 'red'))
        else:
            print(colored('Quadratic model is better', 'green'))

        # check if curr file num is in random -> if yes - add subplot
        if fileNum in rnd:
            plt.subplot(2, 2, plots_to_show)
            plt.plot(x_test, y_predict_linear, '-', c='green', label='linear')
            plt.plot(x_test, y_predict_quadratic, 'ro', label='quadric')
            plt.plot(x_test, y_test, 'bo', c='black', label='actual')
            plt.legend(loc='upper left')
            plt.title(f.title())
            plots_to_show += 1
        print('--------------------------------')

    plt.show()


def parse_arguments():
    """
    This is a function that parses arguments from command line.

    :param: None
    :returns: Namespace storing all arguments from command line
    """
    parser = argparse.ArgumentParser(
        description='Regression')
    parser.add_argument('-m',
                        '--mandatory_argument',
                        type=str,
                        required=True,
                        help='Mandatory - path to dir containing datasets')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
