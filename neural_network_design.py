"""
In order to decide how many hidden nodes the hidden layer should have,
split up the data set into training and testing data and create networks
with various hidden node counts (5, 10, 15, ... 45), testing the performance
for each.

The best-performing node count is used in the actual system. If multiple counts
perform similarly, choose the smallest count for a smaller network with fewer computations.
"""

import numpy as np
from code.myOcr import OCR
from code.ocr import OCRNeuralNetwork
from sklearn.cross_validation import train_test_split


def test(data_matrix, data_labels, test_indices, nn):
    avg_sum = 0
    for j in range(10):
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 10


# Load data samples and labels into matrix
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter=',').tolist()
data_labels = np.loadtxt(open('dataLabels.csv', 'rb')).tolist()

# Create training and testing sets.
train_indices, test_indices = train_test_split(list(range(5000)))

print("PERFORMANCE")
print("-----------")


def test_hidden_layer_size():
    for num_nodes_hidden_layer in range(1, 50):
        nn = OCR(hidden_layer_size=num_nodes_hidden_layer, data=data_matrix,
                 correct_digit_for_data=data_labels, training_indices=train_indices)
        # nn = OCRNeuralNetwork(num_nodes_hidden_layer, data_matrix, data_labels, train_indices, False)
        performance = test(data_matrix, data_labels, test_indices, nn)
        # print("{i} Hidden Nodes: {val}".format(i=num_nodes_hidden_layer, val=performance))
        print("{i},{val}".format(i=num_nodes_hidden_layer, val=round(performance, 3)))


def test_learning_rate():
    for rate in range(2, 100, 2):
        nn = OCR(hidden_layer_size=20, data=data_matrix,
                 correct_digit_for_data=data_labels, training_indices=train_indices,
                 learning_rate=rate/100)
        # nn = OCRNeuralNetwork(num_nodes_hidden_layer, data_matrix, data_labels, train_indices, False)
        performance = test(data_matrix, data_labels, test_indices, nn)
        # print("{i} Hidden Nodes: {val}".format(i=num_nodes_hidden_layer, val=performance))
        print("{i},{val}".format(i=rate, val=round(performance, 3)))


# Try various number of hidden nodes and see what performs best
# test_hidden_layer_size()
test_learning_rate()