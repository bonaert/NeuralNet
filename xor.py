import random
from NeuralNet import NeuralNet

data = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0
}

TRAINING_SAMPLES = 20000

network = NeuralNet(input_size=2, hidden_layer_size=3, output_size=1, learning_rate=0.75)

# Step 1: training
samples = list(data.items())
errors = []
for i in range(TRAINING_SAMPLES):
    neural_net_input, result = random.choice(samples)
    error = network.train(neural_net_input, result)
    errors.append(abs(error))

# Step 2: test
final_errors = []
for neural_net_input, result in data.items():
    prediction = network.predict(neural_net_input)[0]
    print("Input: ", neural_net_input, " -> Output: ",prediction)
    final_errors.append(abs(result - prediction))

avg_error = sum(final_errors) / len(final_errors)
print("Average error: ", avg_error)

import matplotlib.pyplot as plt
plt.plot(errors)
plt.show()