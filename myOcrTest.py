import random
import unittest
from code.myOcr import OCR
from code.NeuralNet import NeuralNet


class OCRTest(unittest.TestCase):
    def make_expected_result_list_for_digit(self, correct_digit):
        vals = [0] * 10
        vals[correct_digit] = 1
        return vals

    def get_max_index(self, values):
        maximum = max(values)
        return values.index(maximum)

    @unittest.skip("la")
    def test_example(self):
        ocr = OCR(restore_from_file=False, input_size=2, hidden_layer_size=3, output_size=2)

        input_data = [0.05, 0.10]
        ocr.predict(input_data)
        data = [(input_data, 1)]
        ocr.train_samples(data)

        self.assertEqual(True, True)

    def test_simple(self):
        ocr = NeuralNet(restore_from_file=False, input_size=2, hidden_layer_size=3, output_size=10)
        input_data = [5, 3]
        correct_digit = 8
        result = self.make_expected_result_list_for_digit(correct_digit)
        for i in range(100):
            ocr.train(input_data, result)

        values = ocr.predict(input_data)
        nn_result = self.get_max_index(values)
        self.assertEquals(correct_digit, nn_result)

    def random_order(self, *args):
        total = []
        for arg in args:
            total += arg
        random.shuffle(total)
        return total

    def manual_ocr(self):
        data = [
            [[0.15, 0.25],
             [0.20, 0.30]],  # Weights 1
            [[0.40, 0.50],
             [0.45, 0.55]],  # Weights 2
            [0.35, 0.35],  # Bias 1
            [0.60, 0.60]  # Bias 2
        ]
        result = [0.01, 0.99]
        input_data = [0.05, 0.10]
        ocr = NeuralNet(input_size=2, hidden_layer_size=2, output_size=2, data=data, learning_rate=0.5)
        ocr.predict(input_data)
        ocr.train(input_data, result)
        print(result)
        self.assertEqual(1, 1)

    def test_double(self):
        """
        The training order is crucial
        http://stackoverflow.com/questions/8101925/effects-of-randomizing-the-order-of-inputs-to-a-neural-network
        """
        ocr = OCR(restore_from_file=False, input_size=2, hidden_layer_size=30, output_size=10)
        num_training_samples = 500
        data_point1, result_1 = [0, 0], 5
        data_point2, result_2 = [1, 1], 9
        data1 = [(data_point1, result_1) for _ in range(num_training_samples)]
        data2 = [(data_point2, result_2) for _ in range(num_training_samples)]
        data = self.random_order(data1, data2)

        for input_data, correct_digit in data:
            ocr.train_sample(input_data, correct_digit)

        self.assertEquals(ocr.predict(data_point1), result_1)
        self.assertEquals(ocr.predict(data_point2), result_2)


if __name__ == '__main__':
    unittest.main()
