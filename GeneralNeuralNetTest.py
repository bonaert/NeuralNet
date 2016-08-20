import unittest
from GeneralNeuralNet import GeneralNeuralNet


class MyTestCase(unittest.TestCase):
    def test_creation(self):
        neural_net = GeneralNeuralNet(layer_sizes=[99, 20, 5, 4, 4, 2, 1, 10, 5])
        self.assertTrue(True)

    def test_exceptions(self):
        self.assertRaises(Exception, lambda: GeneralNeuralNet(layer_sizes=[]))
        self.assertRaises(Exception, lambda: GeneralNeuralNet(layer_sizes=[-1]))
        self.assertRaises(Exception, lambda: GeneralNeuralNet(layer_sizes=[5, 0, 10]))
        self.assertRaises(Exception, lambda: GeneralNeuralNet(layer_sizes=[5, -1, 10]))
        self.assertRaises(Exception, lambda: GeneralNeuralNet(layer_sizes=[5, 100000, 10]))

    def test_prediction(self):
        neural_net = GeneralNeuralNet(layer_sizes=[5, 7, 10, 5, 6])
        neural_net.predict([0.7, 0.5, 0.3, 0.8, 0.2])
        self.assertTrue(True)

    def test_training_and_save(self):
        neural_net = GeneralNeuralNet(layer_sizes=[5, 7, 10, 5, 3], filename='general.txt')
        neural_net.train([0.7, 0.5, 0.3, 0.8, 0.2], [0.1, 0.2, 0.9])
        neural_net.save_data()
        neural_net = GeneralNeuralNet(restore_from_file=True, filename='general.txt', layer_sizes=[5, 7, 10, 5, 3])
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
