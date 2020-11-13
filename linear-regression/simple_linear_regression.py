import numpy as np
import matplotlib.pyplot as plt


### b_1 = sum of cross-deviations /  the sum of squared deviations
### b_0 = mean_y - b_1 * mean_x
### sum of cross-deviations = E(i=1 to n)y_i * x_i - n * mean_x * mean_y
### sum of squared deviations = E(i=1 to n) x_i * x_i - n(mean_x * mean_x)
def coefficient(x, y):
    n = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    sum_of_cross_deviation = np.sum(x * y) - n * mean_x * mean_y
    sum_of_square_deviation = np.sum(x * x) - n * (mean_x * mean_x)

    b_1 = sum_of_cross_deviation / sum_of_square_deviation
    b_0 = mean_y - b_1 * mean_x
    return (b_0, b_1)


def show_plot(x, y, b):
    plt.scatter(x, y, color="b", marker="o", s=30)
    plt.plot(x, b)
    plt.xlabel('Input Data')
    plt.ylabel('Output Data')
    plt.show()


### return np array
def linear_regression(x, b_0, b_1):
    return b_0 + b_1 * x


def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    ### calculate coeffient values
    b_0, b_1 = coefficient(x, y)

    ### calculate linear equation
    regression_line = linear_regression(x, b_0, b_1)

    ### show on plot
    show_plot(x, y, regression_line)


if __name__ == '__main__':
    main()
