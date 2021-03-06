import json
import numpy
import utils
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


class GeneralNeuralNet:
    MAX_LAYER_SIZE = 100

    def __init__(self, restore_from_file=False, filename=None,
                 layer_sizes=None, learning_rate=0.4, momentum=0.3, data=None):
        self.FILE_PATH = filename or "data.txt"
        self.LEARNING_RATE = learning_rate
        self.MOMENTUM = momentum
        self.layer_sizes = layer_sizes

        self.num_matrices = len(layer_sizes) - 1
        self.weights_changes = []
        self.bias_changes = []

        self.inputs = []
        self.net_inputs = []
        self.outputs = []
        self.errors = []
        self.grad_weights = []
        self.grad_bias = []

        self.set_up_matrices(data, restore_from_file)

        self.sigmoid = numpy.vectorize(self._sigmoid_scalar)
        self.sigmoid_prime = numpy.vectorize(self._sigmoid_prime_scalar)

    def set_up_matrices(self, data, restore_from_file):
        if self.layer_sizes is None or len(self.layer_sizes) == 0:
            raise Exception("The list of sizes must not be empty or None.")

        are_sizes_invalid = any([not (0 < size < self.MAX_LAYER_SIZE) for size in self.layer_sizes])
        if are_sizes_invalid:
            raise Exception("The sizes must be between 1 and %s" % self.MAX_LAYER_SIZE)

        self._initialize_matrices(data, restore_from_file)

    def _initialize_matrices(self, data, restore_from_file):
        if restore_from_file:
            file_data = self.retrieve_data()
            if file_data is not None:
                self._set_matrices(file_data)
                return

        if data is None:
            self._initialize_weights_randomly()
        else:
            self._set_matrices(data)

    def _set_matrices(self, file_data):
        bias_list = file_data["bias"]
        matrix_weight_list = file_data["weights"]
        num_layers_in_file_data = len(matrix_weight_list) + 1

        if self.get_num_layers() != num_layers_in_file_data:
            raise Exception("The number of layers in the file is different from the number of layers you specified.")

        self.bias_list = []
        self.matrix_weights_list = []
        for i in range(self.get_num_layers() - 1):
            input_size, output_size = self.layer_sizes[i], self.layer_sizes[i + 1]
            bias = bias_list[i]
            matrix_weights = matrix_weight_list[i]

            bias_size = len(bias)
            if bias_size != output_size:
                raise Exception("The bias for layer %d should have %d elements but has %d (on file)" % (i, output_size,
                                                                                                        bias_size))

            num_rows = len(matrix_weights)
            if num_rows != input_size:
                raise Exception("The weights matrix %d should have %d rows but has %d rown on file" % (i, input_size,
                                                                                                       num_rows))
            num_columns = len(matrix_weights[0])
            if num_columns != output_size:
                raise Exception(
                    "The weights matrix %d should have %d rows but has %d rows on file" % (i, output_size, num_columns))

            numpy_matrix_weights = numpy.mat([numpy.array(li) for li in matrix_weights])
            numpy_bias = numpy.mat([numpy.array(bias)])
            self.bias_list.append(numpy_matrix_weights)
            self.matrix_weights_list.append(numpy_bias)

    def _sigmoid_scalar(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_prime_scalar(self, x):
        sigmoid_val = self._sigmoid_scalar(x)
        return sigmoid_val * (1 - sigmoid_val)

    def _initialize_weights_randomly(self):
        self.bias_list = []
        self.matrix_weights_list = []
        for i in range(self.num_matrices):
            input_size, output_size = self.layer_sizes[i], self.layer_sizes[i + 1]

            self.bias_list.append(self._make_random_matrix_with_small_numbers(1, output_size))
            self.matrix_weights_list.append(self._make_random_matrix_with_small_numbers(input_size, output_size))

    def get_num_layers(self):
        return len(self.layer_sizes)

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

        # If we're at layer X, there's no need to keep track of the result of the previous layers
        # so we simply replace the current state each time we go to the next layer
        current_state = input_data
        for (weights_matrix, bias) in zip(self.matrix_weights_list, self.bias_list):
            current_state = numpy.dot(current_state, weights_matrix)
            current_state += bias
            current_state = self.sigmoid(current_state)

        return current_state.tolist()[0]

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
        :param input_data: the input of the training example
        :param result: the output of the training example
        """
        input_data = numpy.mat(input_data)

        # Step 1: use our current neural network to predict an outcome
        self.inputs = []
        self.net_inputs = []
        self.outputs = []

        current_layer = input_data
        for (weights_matrix, bias) in zip(self.matrix_weights_list, self.bias_list):
            self.inputs.append(current_layer)
            net_input_next_layer = numpy.dot(current_layer, weights_matrix) + bias
            self.net_inputs.append(net_input_next_layer)

            current_layer = output = self.sigmoid(net_input_next_layer)
            self.outputs.append(output)

        # Step 2: compute the error term (dE/dOuput) for each layer
        output_layer_errors = self.outputs[-1] - numpy.mat(result)

        self.errors = [output_layer_errors]
        for i in range(self.num_matrices - 1, 0, -1):
            error_next_layer = self.errors[0]
            net_input_next_layer = self.net_inputs[i]
            weights_next_layer = self.matrix_weights_list[i]

            sigmoid_prime_next_layer = self.sigmoid_prime(net_input_next_layer)
            adjusted_next_layer_errors = numpy.multiply(error_next_layer, sigmoid_prime_next_layer)
            error_current_layer = numpy.dot(adjusted_next_layer_errors, weights_next_layer.T)
            self.errors.insert(0, error_current_layer)

        # Step 3: compute the gradient of the weights (and bias) on the error
        self.grad_weights = []
        self.grad_bias = []

        for i in range(self.num_matrices - 1, -1, -1):
            errors = self.errors[i]
            net_input = self.net_inputs[i]
            input = self.inputs[i]

            grad_weights = numpy.dot(input.T, errors)
            sigmoid_prime_layer = self.sigmoid_prime(net_input)

            gradient_weights = numpy.multiply(grad_weights, sigmoid_prime_layer)
            gradient_bias = numpy.multiply(sigmoid_prime_layer, errors)

            self.grad_bias.insert(0, gradient_bias)
            self.grad_weights.insert(0, gradient_weights)

        # Step 4: update the weights
        for i in range(self.num_matrices):
            self.matrix_weights_list[i] -= self.LEARNING_RATE * self.grad_weights[i]
            self.bias_list[i] -= self.LEARNING_RATE * self.grad_bias[i]

        # Step 5: Add momentum
        if self.weights_changes:
            for i in range(self.num_matrices):
                self.matrix_weights_list[i] -= self.MOMENTUM * self.weights_changes[i]
                self.bias_list[i] -= self.MOMENTUM * self.bias_changes[i]

        self.weights_changes = self.grad_weights
        self.bias_changes = self.grad_bias

    def save_data(self):
        data = {
            "bias": [],
            "weights": []
        }
        for i in range(len(self.matrix_weights_list)):
            weight_matrix = [np_mat.tolist()[0] for np_mat in self.matrix_weights_list[i]]
            bias = self.bias_list[i].tolist()[0]
            data["bias"].append(bias)
            data["weights"].append(weight_matrix)

        # data = {
        #     "matrix_weights1": ,
        #     "matrix_weights2": [np_mat.tolist()[0] for np_mat in self.matrix_weights2],
        #     "bias1": self.bias1[0].tolist()[0],
        #     "bias2": self.bias2[0].tolist()[0]
        # }
        with open(self.FILE_PATH, 'w') as f:
            f.write(json.dumps(data))

    def retrieve_data(self):
        if utils.file_exists(self.FILE_PATH):
            with open(self.FILE_PATH) as f:
                try:
                    return json.load(f)
                except (ValueError, KeyError):
                    return None
