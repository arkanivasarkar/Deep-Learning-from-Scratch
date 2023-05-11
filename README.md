# Neural-Networks-from-Scratch

This repository contains the implementation of deep learning networks from scratch.

&nbsp;
## Description
This project was developed as a part of the programming exercises of the **Deep Learning** course offered by the [**Pattern Recognition Lab**](https://lme.tf.fau.de/) at **Friedrich-Alexander-Universität (FAU)**.

The codes are written in *Python* using object oriented programming concepts such as, *inheritance* or *polymorphism*. All fundamental layers, activation and loss functions, optimizers and regularizers are implemented by coding the corresponding mathematical operations using *NumPy* only, without the use of any deep learning frameworks. 

&nbsp;
&nbsp;
## Methods
The project is implemented in three parts as mentioned below:

    Part 1:    Feed-forward Neural Network
            
 A `fully connected layer` object and `ReLU` & `Softmax` activations were developed as well as the `Cross Entropy` loss.

2. **Part 2:** Convolutional Neural Network
     Basic blocks of Convolutional Neural Networks were devloped (`Conv` layer and `Pooling`). Several optimizers such as `SGD with Momentum` and `ADAM` were also developed.

3. **Phase 3 (Recurrent layer & Regularization):** The classical `LSTM` Unit layer (basically the most used RNN architecture to date), which can be used in Recurrent Neural Networks, was developed (including more activations such as `TanH` or `Sigmoid`. Also different regularization layers (like `Batch Normalization`, `Dropout`, `L1` and `L2` regulizers) were developed.

The main class running everything is `NeuralNetwork.py`. Various unit tests for every layer and function are included in `NeuralNetworkTests.py`.

Further details such as task descriptions and specifications can be found inside *./Protocols/* directory.

### Miscellaneous
In the `./Generator` directory, you can find an image generator which returns the input of a neural network each time it gets called.
This input consists of a batch of images with different augmentations and its corresponding labels. 100 sample images are provided in the `./Generator/Sample_data` to test this generator.
