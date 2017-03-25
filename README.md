# Logistic Regression With Gradient Descent & Newton's Method
Train and test a Logistic Regression model using Gradient Descent and Newton's Method.

## Requirements:
* GNU Octave

## The Model:
* Binary classifier (can be adapted for multiclass)
* Uses the regularized cross-entropy error function

## The Dataset:
* From the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Ionosphere).
* Contained in ```.dat``` files in the ```/ionosphere``` directory.

## Source Files:
* ```main.m```: contains demo of Gradient Descent and Newton's Method on Ionosphere dataset.
* ```gradient_descent.m```: trains logistic model using Batch Gradient Descent and returns two variables, w, the weight vector, and b the bias term.
* ```newton.m```: trains logistic model using Newton's Method, returning same variables as above.
* ```testLogisticRegression.m```: reports the accuracy of the logistic model.

## Auxiliary Files and Functions:
* ```sigmoid.m```: computes Sigmoid of given value.
* ```bgradient.m```: computes gradient with respect to the bias term
* ```wgradient.m```: computes gradient with respect to the weight vector

## Instructions:
* Run ```main``` in Octave.