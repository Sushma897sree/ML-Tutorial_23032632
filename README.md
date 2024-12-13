# ML-Tutorial_23032632
# **Multilayer Perceptron (MLP) for Image Classification: Fashion-MNIST**

## Overview
This project demonstrates the implementation of a Multilayer Perceptron (MLP) model on the Fashion-MNIST dataset. The project includes a tutorial on the theory behind MLPs, activation functions, backpropagation, optimization techniques, and evaluation metrics. It shows how MLPs can be used for image classification tasks, including practical techniques such as regularization, learning rate scheduling, early stopping, and hyperparameter tuning.


---------------------------------------------------------------------------------------------------------------------------------------------------------------
## Project Structure
**fashion_mnist_mlp_model.ipynb:** Jupyter notebook implementing the MLP model on the Fashion-MNIST dataset.
README.md: This file.

**Tutorial Document (MLP_Tutorial.pdf):** A detailed tutorial explaining the MLP architecture and how it applies to Fashion-MNIST.

**References:** Academic papers, articles, and sources related to MLP and neural network theory.


### Getting Started
#### Prerequisites
Make sure you have the following libraries installed:

Python 3.x

TensorFlow (v2.x)

Keras

Matplotlib

NumPy

Install the required libraries using the following:

bash

`pip install tensorflow numpy matplotlib`

#### Dataset
The Fashion-MNIST dataset contains 60,000 training images and 10,000 test images, each of size 28x28 pixels. It includes 10 categories like T-shirts, trousers, and sneakers, commonly used for benchmarking image classification models.

python

`from tensorflow.keras import datasets
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()`

## *1. Load and Preprocess Data*
python

#### Normalize the images to values between 0 and 1
`X_train, X_test = X_train / 255.0, X_test / 255.0`

## *2. Build the Model*

python

`from tensorflow.keras import layers, models`

### Building the MLP model
```
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the images into vectors
    layers.Dense(128, activation='relu'),  # Hidden layer with ReLU activation
    layers.Dropout(0.2),  # Dropout for regularization to prevent overfitting
    layers.Dense(10, activation='softmax')  # Output layer for classification (Softmax)
])

```



## *Display model summary*
`model.summary()`

## *Compile the Model*

python

``` model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

## *Train the Model*

python

`history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))`

## *Evaluate the Model*

python

``` test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
```

## *Advanced Techniques Implemented*

*1. Regularization Techniques:*
Dropout: Randomly deactivates neurons during training to prevent overfitting.
L2 Regularization: Penalizes large weights to reduce overfitting.

*2. Early Stopping:*
Stops the training process if the validation loss does not improve for a specified number of epochs (patience=3).

*3. Learning Rate Scheduling:*
Adjusts the learning rate dynamically during training to optimize convergence.

## *Evaluation Metrics*
*Accuracy:* Percentage of correct predictions.

*Precision, Recall, F1-Score:* Measures of model performance, especially for imbalanced datasets.

*Confusion Matrix:* Compares actual vs. predicted classes to understand the model's errors.

*Training/Validation Curves:* Helps identify overfitting or underfitting during training.

## *References*
•	Aggarwal, C. (2018). Neural Networks and Deep Learning: A Textbook. Springer.

•	DataCamp. (2020). Mastering Backpropagation: A Comprehensive Guide for Neural Networks. Retrieved from https://www.datacamp.com.

•	Scikit-learn Documentation. (2021). Varying Regularization in Multi-layer Perceptron. Retrieved from https://scikit-learn.org.

•	Chollet, F. (2017). Deep Learning with Python. Manning Publications.

•	Towards Data Science. (2020). Understanding the Basics of Multi-layer Perceptrons (MLPs). Retrieved from https://towardsdatascience.com


## *License*
This project is licensed under the MIT License. See the *LICENSE* file for details.

### How to Contribute
Feel free to fork this repository, open issues, or submit pull requests. Contributions are welcome!

