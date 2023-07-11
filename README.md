# keras-img-classification
Image classification with Python's Keras using my travelling photos

My blog article about this project: [here](https://rolkotech.blogspot.com/2023/07/image-classification-with-pythons-keras.html)

## Description

In this project I use my own travelling photos to train a neural network so that I can classify my photos. Currently, the model classifies the images based on if they contain water (sea, river, lake, etc) or not.
<br>
Also, I created a library to be able to classify images manually and quickly. This is required to create the training and validation dataset for our model.
<br>
I use Python's Keras (tensorflow) library.

I go through the following steps of an image classification project:
* Creating a classified image dataset using our own photos
* Loading the image dataset
* Sampling the data
* Normalizing the data (rescaling)
* Building and training the neural network
* Saving our model
* Loading a previously saved model
* Predict categories of unseen images
* Visualization techniques for sampling and prediction

Files:
* `classify_images.py` : library to classify images manually and quickly
* `tfk_classify_water.py` : library that builds and trains the neural network using Keras
* `tfk_predict_water.py` : library that uses our model to predict image categories

Third-party libraries used:
* tensorflow (keras)
* numpy
* matplotlib
* opencv-python

Python version used for the development: Python 3.9.6
