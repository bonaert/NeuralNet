import os
import json
import utils
import numpy
from math import exp
from random import random

"""

######################### DIMENSIONS ################################

Here are the different matrices:
self.matrix_weights1 -> INPUT_SIZE * HIDDEN_LAYER_SIZE (takes in input, outputs input for hidden layer)
self.matrix_weights2 -> HIDDEN_LAYER_SIZE * OUTPUT_SIZE (takes in hidden layer input, output 10 values)

Input data: the squares of the images (0=not painted, 1=painted)
To get the result of the first layer, we apply the weights and add the bias.

Therefore, res = input * matrix_weights1 + bias1
input * matrix_weights1 -> (1 x INPUT_SIZE) * (INPUT_SIZE x HIDDEN_LAYER_SIZE) = 1 x HIDDEN_LAYER_SIZE

If we want to add the bias, it must have the right dimensions -> 1 x HIDDEN_LAYER_SIZE
Then, we apply the sigmoid function on every element.

Similarly, for the second layer, res = output * matrix_weights2 + bias2
output * matrix_weights2 -> (1 x HIDDEN_LAYER_SIZE) * (HIDDEN_LAYER_SIZE x OUTPUT_SIZE) = 1 x OUTPUT_SIZE

If we want to add the bias, it must have the right dimensions -> 1 x OUTPUT_SIZE

################### BACK PROPAGATION ###################################

Here we have first to compute the errors. Since the dimensions of the output are 1 x OUTPUT_SIZE, then
the errors for the last layer too (1 x OUTPUT_SIZE)

To add the weight corrections, the output must have the same dimensions as the weight matrix, because
correction is done by adding the output (multiplied by a learning rate) to the weight matrix.

Therefore, the 2 weight correction matrices must have as dimensions
corrections_mat2 = HIDDEN_LAYER_SIZE x OUTPUT_SIZE
corrections_mat1 = INPUT_SIZE x HIDDEN_LAYER_SIZE

The bias have as dimensions, 1 x HIDDEN_SIZE, 1 x OUTPUT_SIZE, like the errors, so the correction
is simple (just multiply by the learning rate and add to the bias)

Errors: 1 x OUTPUT_SIZE
Input for the 2nd layer: 1 x HIDDEN_LAYER_SIZE
corrections_mat2 -> (HIDDEN_LAYER_SIZE x 1) * (1 x OUTPUT_SIZE) = HIDDEN_LAYER_SIZE x OUTPUT_SIZE
Therefore, corrections_mat2 = input.T * errors

We then need to add the sigmoid prime factor, which applies for each node in the result.
The number of nodes in that layer is OUTPUT_SIZE, and the correction matrix is
HIDDEN_LAYER_SIZE x OUTPUT_SIZE, therefore we must correct each column by sigmoid_prime(z_i).

Therefore, it should be (input.T * errors) x (sigmoid) for the last layer.
In the first layer, the only difference is how we compute the errors, which is more complicated.
To compute the errors, we must see how the node affects the cost. Since we can't do this directly, we
need to use a trick.

The idea is to see how the node affects the nodes in the next layer, and how those nodes affect the
cost (in a recursive fashion). By combining the two (via the chain rule), we can compute the cost accurately.

The maths is the following:
dCost_i/dWeight = sum(dCost_i/dOutput_f * dOutput_f/dTotal_input_f * dTotal_input_f/dOutput_node)
for all nodes f in the following layer.
In other words, dCost_i/dWeight = sum(output_errors_f * sigmoid_prime(f) * weight_f)

Again, this must have the appropriate dimensions (similar to the other error matrix): 1 x HIDDEN_LAYER_SIZE

Weights: HIDDEN_LAYER_SIZE x OUTPUT_SIZE
Output errors: 1 x OUTPUT_SIZE

Therefore, res = (1 x OUTPUT_SIZE) * (OUTPUT_SIZE x HIDDEN_LAYER_SIZE) = 1 x HIDDEN_LAYER_SIZE
           res = output_errors * weights.T

Afterwards, we need to apply the sigmoid prime function, in the same fashion. Again, we must
multiply by the sigmoid_prime(input_f) element_by_element.

Therefore, hidden_errors = (output_errors * weights.T) x sigmoid_prime(output_layer_2)
"""


class NeuralNet:
    def __init__(self, restore_from_file=False, filename=None,
                 input_size=20 * 20, hidden_layer_size=20, output_size=10, learning_rate=0.1,
                 data=None, correct_digit_for_data=None, training_indices=None):
        self.FILE_PATH = filename or "data.txt"
        self.LEARNING_RATE = learning_rate
        self.INPUT_SIZE = input_size
        self.HIDDEN_LAYER_SIZE = hidden_layer_size
        self.OUTPUT_SIZE = output_size
        self.epochs = 0

        self._initialize_matrices(restore_from_file)
        self.sigmoid = numpy.vectorize(self._sigmoid_scalar)
        self.sigmoid_prime = numpy.vectorize(self._sigmoid_prime_scalar)


        if (not os.path.isfile(self.FILE_PATH) or not restore_from_file):
            # Train using sample data
            if training_indices:
                for i in training_indices:
                    self.train(data[i], correct_digit_for_data[i])
                self.save_data()



    def _initialize_matrices(self, restore_from_file):
        if restore_from_file:
            file_data = self.retrieve_data()
            if file_data is not None:
                self._set_matrices(file_data)
                return
        else:
            self._initialize_weights_randomly()

    def _set_matrices(self, file_data):
        matrix_weights1, matrix_weights2, bias1, bias2 = file_data

        self.matrix_weights1 = numpy.mat([numpy.array(li) for li in matrix_weights1])
        self.matrix_weights2 = numpy.mat([numpy.array(li) for li in matrix_weights2])
        self.bias1 = numpy.mat([numpy.array(bias1)])
        self.bias2 = numpy.mat([numpy.array(bias2)])

    def _sigmoid_scalar(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_prime_scalar(self, x):
        sigmoid_val = self._sigmoid_scalar(x)
        return sigmoid_val * (1 - sigmoid_val)

    def _initialize_weights_randomly(self):
        self.bias1 = self._make_random_matrix_with_small_numbers(1, self.HIDDEN_LAYER_SIZE)
        self.bias2 = self._make_random_matrix_with_small_numbers(1, self.OUTPUT_SIZE)
        self.matrix_weights1 = self._make_random_matrix_with_small_numbers(self.INPUT_SIZE, self.HIDDEN_LAYER_SIZE)
        self.matrix_weights2 = self._make_random_matrix_with_small_numbers(self.HIDDEN_LAYER_SIZE, self.OUTPUT_SIZE)

    def _make_random_matrix_with_small_numbers(self, size_in, size_out):
        matrix = [[-0.06 + 0.12 * random() for i in range(size_out)] for j in range(size_in)]
        return numpy.mat(matrix)

    def predict(self, input_data):
        """
        Makes a prediction from the given input data.
        :param input_data: the input data
        :return: number: the recognized number
        """
        input_data = numpy.mat(input_data)

        layer1_res = numpy.dot(input_data, self.matrix_weights1)
        layer1_res += self.bias1
        output_layer1 = self.sigmoid(layer1_res)

        layer2_res = numpy.dot(output_layer1, self.matrix_weights2)
        layer2_res += self.bias2
        output_layer2 = self.sigmoid(layer2_res)

        values = output_layer2.tolist()[0]
        return values

    def train_samples(self, data):
        """
        Trains the neural network given a series of inputs and the correct results.
        :param data: a list of tuples containing the input_data and the result the neural net should predict
        :return: None
        """
        for input_data, result in data:
            self.train(input_data, result)

    def train(self, input_data, result):
        """
        Uses the backpropagation algorithm to improve the weights on the network.
        :param input_data: the data from the image
        :param result: the correct digit (that should be predicted by the OCR system)
        """
        input_data = numpy.mat(input_data)

        # Step 1: use our current neural network to predict an outcome
        layer1_net_input = numpy.dot(input_data, self.matrix_weights1) + self.bias1
        layer1_output = self.sigmoid(layer1_net_input)

        layer2_net_input = numpy.dot(layer1_output, self.matrix_weights2) + self.bias2
        layer2_output = self.sigmoid(layer2_net_input)

        # Step 2: compute the errors for each layer
        layer2_errors = -(numpy.mat(result) - layer2_output)

        sigmoid_prime_layer2 = self.sigmoid_prime(layer2_net_input)
        adjusted_output_layer_errors = numpy.multiply(layer2_errors, sigmoid_prime_layer2)
        layer1_errors = numpy.dot(adjusted_output_layer_errors, self.matrix_weights2.T)

        # Step 3: compute the weight changes for each layer
        transposed_layer1_output = layer1_output.T
        grad_weights2 = numpy.dot(transposed_layer1_output, layer2_errors)
        gradient_weights2 = numpy.multiply(grad_weights2, sigmoid_prime_layer2)

        # Todo: check matrix sizes. Do example by hand. See how they should improve. Turn off screen while doing it.
        grad_bias2 = numpy.multiply(sigmoid_prime_layer2, layer2_errors)

        grad_weights1 = numpy.dot(input_data.T, layer1_errors)
        sigmoid_prime_layer1 = self.sigmoid_prime(layer1_net_input)
        gradient_weights1 = numpy.multiply(grad_weights1, sigmoid_prime_layer1)

        grad_bias1 = numpy.multiply(sigmoid_prime_layer1, layer1_errors)

        # Step 4: change the weights and bias
        self.matrix_weights2 -= self.LEARNING_RATE * gradient_weights2
        self.matrix_weights1 -= self.LEARNING_RATE * gradient_weights1
        self.bias2 -= self.LEARNING_RATE * grad_bias2
        self.bias1 -= self.LEARNING_RATE * grad_bias1

        self.epochs += 1

        return numpy.sum(layer2_errors)

    def save_data(self):
        data = {
            "matrix_weights1": [np_mat.tolist()[0] for np_mat in self.matrix_weights1],
            "matrix_weights2": [np_mat.tolist()[0] for np_mat in self.matrix_weights2],
            "bias1": self.bias1[0].tolist()[0],
            "bias2": self.bias2[0].tolist()[0]
        }
        with open(self.FILE_PATH, 'w') as f:
            f.write(json.dumps(data))

    def retrieve_data(self):
        if utils.file_exists(self.FILE_PATH):
            with open(self.FILE_PATH) as f:
                try:
                    data = json.load(f)
                    return data["matrix_weights1"], data["matrix_weights2"], data["bias1"], data["bias2"]
                except (ValueError, KeyError):
                    return None
