# NeuralNet

**Note:** this is an old repository I created when I was first learning about neural network and wanted to be sure I understood them well. Given the current popularity of Neural Networks, if want to learn then you're in luck because there are tons of amazing educational resources (such as Michael Nielsen's [Neural networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) or the [Deep Learning book](http://www.deeplearningbook.org/) by Goodfellow, Bengio and Courville) and great frameworks to play with, such as [Tensorflow](https://www.tensorflow.org/), [Keras](https://keras.io/) or [PyTorch](https://pytorch.org/).

Want to understand how neural nets work, and see a simple implementation? I did, which is why I created this simple and (hopefully) clear implementation of a neural network. It greatly improved my understanding of how they work.

# Blog posts with explanations

This repository is the companion to my [blog posts about learning how neural networks work and learn](http://blog.gregbonaert.com/starter-guide-to-neural-networks-part-1/). There I write the theory behind them, and the maths in the clearest way I could. I don't oversimplify things or make things too vague. 

At the end, you'll know the precise maths required to make the neural net work.

![The structure of the simplest neural network](http://blog.gregbonaert.com/wp-content/uploads/2016/08/NeuralNetSimple-300x205.jpeg)

# The Neural Net Code

* A simple [2 layer Neural Net](https://github.com/bonaert/NeuralNet/blob/master/NeuralNet.py) (to start learning), then a [n-layer Neural Net](https://github.com/bonaert/NeuralNet/blob/master/GeneralNeuralNet.py), with prediction and training, using the classical backpropagation algorithm (with momentum)

# Examples and applications

Want a couple of example of how to use the code? Here some basic things I've built:

* Let's use neural networks to predict the output of the [XOR function](https://github.com/bonaert/NeuralNet/blob/master/xor.py). It's absurd, but a good illustration.
* Then, I built a simple [simple OCR system](https://github.com/bonaert/NeuralNet/blob/master/ocr.py), which regognizes hand-written digits of the MNIST database

# Meta

I also created a couple of script to better understand how to test and see the performance of different configurations:

* This scripts [tests different designs](https://github.com/bonaert/NeuralNet/blob/master/neural_network_design.py) 
* And if you want to visualize the results, you can [make pretty graphs](https://github.com/bonaert/NeuralNet/blob/master/show_graph.py)

