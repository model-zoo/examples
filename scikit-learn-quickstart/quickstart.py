"""
Scikit Learn: Quick Start
=========================

In this tutorial, we are going to train and deploy a small scikit-learn
classifier on the [iris
dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html),
and deploy it to Model Zoo to make predictions via HTTP.

You can follow along this tutorial in any Python environment you're comfortable
with, such as a Python IDE, Jupyter notebook, or a Python terminal. The easiest
option is to open this tutorial directly in colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/model-zoo/examples/blob/master/scikit-learn-quickstart/quickstart.ipynb)

Installation
------------

Install the Model Zoo client library via pip:
"""

#!pip install modelzoo-client[sklearn]

##############################################################################
#
# To deploy and use your own models, you'll need to create an account and
# configure an API key. You can do so from the command line:

#!modelzoo auth

##############################################################################
#
# Train
# ----
#
# First, we need to train a model. For the sake of this quickstart, we'll train
# a simple logistic regression model on the [iris
# dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection

# Load and split data into train and test sets.
iris = sklearn.datasets.load_iris()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    iris.data, iris.target, test_size=0.1
)

# Train the logistic regression model.
estimator = sklearn.linear_model.LogisticRegression()
estimator.fit(X_train, y_train)

##############################################################################
#
# Deploy
# ------
#
# To deploy this pipeline to a production-ready HTTP endpoint, use the
# [`modelzoo.sklearn.deploy()`](https://docs.modelzoo.dev/reference/modelzoo.sklearn.html#modelzoo.sklearn.deploy)
# function. This function will directly take any scikit learn estimator object.

import modelzoo.sklearn

model_name = modelzoo.sklearn.deploy(estimator)

##############################################################################
#
# That's all there is to it! Behind the scenes, Model Zoo serialized your model
# via
# [`joblib`](https://scikit-learn.org/stable/modules/model_persistence.html),
# uploaded it to object storage, and deployed a serverless lambda function to
# serve requests to this model. If you'd like, take some time to explore the
# model via the Web UI link. There you'll be able to modify documentation, test
# the model with raw or visual inputs, monitor metrics and/or logs. By default,
# only your account (or anybody you share your API key with) will be able to
# access this model.
#
# You can specify the name of the model you'd like to deploy via a ``model_name``
# argument. If a name is omitted, Model Zoo will choose a unique one for you.
# Model names must be unique to your account.
#
# Predict
# -------
#
# Next, we'll use our Python client library to query the model for a
# prediction.
# [`modelzoo.sklearn.predict()`](https://docs.modelzoo.dev/reference/modelzoo.sklearn.html#modelzoo.sklearn.predict)
# requires the `model_name` and an input numpy array for prediction. Under the
# hood, the client library will query the model endpoint for a result. Let's
# use the test set we loaded earlier, and compare the predictions to the ground
# truth labels.

import sklearn.metrics

prediction = modelzoo.sklearn.predict(model_name, X_test)

print("Test label ground truth: ", y_test)
print("Model prediction: ", prediction)
print("Test accuracy: ", sklearn.metrics.accuracy_score(y_test, prediction))

##############################################################################
#
# Great! At this point, we've successfully used our scikit learn classifier to
# make predictions on test data.
#
# Interested in what you've seen and want to test drive an unlimited version of
# Model Zoo? Apply to our [private
# beta](https://modelzoo.typeform.com/to/Y8U9Lw) and reach out at
# [contact@modelzoo.dev](mailto:contact@modelzoo.dev) to learn more.
