# NeuralNetworksHomeAssignment1

Student Information

- Name: Ruthvik Reddy Gaddam
- Student ID: 700755809 RXG58090
- Course: CS5720 Neural Networks CRN-23848

## Overview

This repository contains the final submission of the home assignment of the Neural Networks course. The submission topics include Tensor operations, reshaping, broadcasting, implenting and comparing loss functions (MSE and CCE), training neural nets on mnist dataset with adam and stochastic gradient descent optimizers, training a neural network and logging it to TensorBoard.

- File Structure:
- The assignment contains 4 tasks which were implemented separately in 4 python notebooks
    - ha1task1.ipynb
    - ha1task2.ipynb
    - ha1task3.ipynb
    - ha1task4.ipynb

## How to execute locally

Download the repository and install the mentioned packages in the requirements.txt file and run the cells of the Jupyter Notebook

### 1. Tensor Manipulations & Reshaping

Task: Create a random tensor of shape (4, 6).

Find its rank and shape using TensorFlow functions.

Reshape it into (2, 3, 4) and transpose it to (3, 2, 4).

Broadcast a smaller tensor (1, 4) to match the larger tensor and add them.

Explain how broadcasting works in TensorFlow.

Expected Output:

- Printed rank and shape of the tensor before and after reshaping/transposing.

Result:
- Created a random tensor and found shape and rank of tensor
- Reshaped a tensor to the mentionsed shape
- Added tensors using broadcasting without explicitly reshaping the smaller tensor

## 2. Loss Functions & Hyperparameter Tuning

Task:

- Define true values (y_true) and model predictions (y_pred).
- Compute Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE) losses.
- Modify predictions slightly and check how loss values change.
- Plot loss function values using Matplotlib.

Expected Output:

- Loss values printed for different predictions.

- Bar chart comparing MSE and Cross-Entropy Loss.

Result:

- Defined true and predicted values for label and computed MSE and CCE loss values
- Updated the predicted values ot observe the change in MSE and CCE
- Plotted the loss values as a bar chart


## 3. Train a Model with Different Optimizers

Task: 
- Load the MNIST dataset.
- Train two models: One with Adam and another with SGD
- Compare training and validation accuracy trends.

Expected Output:

- Accuracy plots comparing Adam vs. SGD performance.

Result:
- Loaded the mnist dataset and processed it by normalizing the pixel values and one hot encoded label values
- Two models with adam and SGD optimizers were trained for 5 epochs
- Compared the training and validation accuracy of two models

## 4. Train a Neural Network and Log to TensorBoard

Task:

- Load the MNIST dataset and preprocess it. 
- Train a simple neural network model and enable TensorBoard logging.
- Launch TensorBoard and analyze loss and accuracy trends.

Expected Output:

- The model should train for 5 epochs and store logs in the "logs/fit/" directory.

- You should be able to visualize training vs. validation accuracy and loss in TensorBoard.

Result:
- Loaded the mnist dataset and processed it by normalizing the pixel values and one hot encoded label values
- A simpleneural net was trained for 5 epochs and the logs were stored in "logs/fit" directory ("logs/fir/hw1").
- Visualized training vs. validation accuracy and loss in TensorBoard.
