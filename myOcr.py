from code.NeuralNet import NeuralNet

"""
Uses the neural net service to make a simple OCR system.
The input to the neural net is a matrix of pixels (1=painted, 0=blank) and the output is the most recognized digit.

You can customize the input size and the size of the hidden layer, and the learning rate of the neural network.
If you have trained the neural network, you can save the neural network to a file.

In a later session, you can restore the neural network from the file by using a flag.
"""


class OCR:
    def __init__(self, restore_from_file=False, filename=None,
                 input_size=20 * 20, hidden_layer_size=20, learning_rate=0.4,
                 data=None, correct_digit_for_data=None, training_indices=None):
        result=None
        if data is not None:
            result = []
            for correct_digit in correct_digit_for_data:
                correct_digit_result = self.__make_expected_result_list_for_digit(int(correct_digit))
                result.append(correct_digit_result)

        self.neural_net = NeuralNet(restore_from_file=restore_from_file,
                                    filename=filename,
                                    input_size=input_size,
                                    hidden_layer_size=hidden_layer_size,
                                    output_size=10,
                                    learning_rate=learning_rate,
                                    data=data,
                                    correct_digit_for_data=result,
                                    training_indices=training_indices)


    def save_data(self):
        self.neural_net.save_data()

    def __make_expected_result_list_for_digit(self, correct_digit):
        result = [0] * 10
        result[correct_digit] = 1
        return result

    def train_sample(self, input_data, correct_digit):
        result = self.__make_expected_result_list_for_digit(correct_digit)
        self.neural_net.train(input_data, result)

    def train_samples(self, data):
        for input_data, correct_digit in data:
            self.train_sample(input_data, correct_digit)

    def predict(self, input_data):
        values = self.neural_net.predict(input_data)
        maximum_prob = max(values)
        result = values.index(maximum_prob)
        return result
