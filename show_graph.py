import os
import matplotlib.pyplot as plt


def make_graph(x, y, x_label, y_label, axis_range, save_file_name):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis(axis_range)
    plt.savefig(save_file_name)
    plt.show()


def make_perf_graph():
    x = []
    y = []
    with open("performance", "r") as f:
        for line in f:
            i, performance = line.strip().split(",")
            i, performance = int(i), round(float(performance) * 100)
            x.append(i)
            y.append(performance)
    make_graph(x, y, "Number of nodes in the hidden layer", "OCR success rate", [0, 50, 0, 100], "plot.png")


def make_perf_learning_rate_graph():
    x = []
    y = []
    with open("performance_learning_rate.txt", "r") as f:
        for line in f:
            i, performance = line.strip().split(",")
            i, performance = float(i) / 100, round(float(performance) * 100)
            x.append(i)
            y.append(performance)
    make_graph(x, y, "Learning rate of the neural network (20 node hidden layer)", "OCR success rate", [0, 1, 0, 100],
               "plot_learning_rate.png")


# make_perf_graph()
make_perf_learning_rate_graph()