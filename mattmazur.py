from NeuralNet import NeuralNet

network = NeuralNet(restore_from_file=True, filename="mattmazur.json",
                    input_size=2, hidden_layer_size=2, output_size=2,
                    learning_rate=0.5)
network.predict([0.05, 0.10])
network.train([0.05, 0.10], [0.01, 0.99])

