"""
PyTorch: Quick Start
====================

In this tutorial, we are going to deploy a [PyTorch image classification
model](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) to Model Zoo and
use it to make a prediction. The Model Zoo client library relies on the [ONNX
format](https://onnx.ai/) to serialize models.

You can follow along this tutorial in any Python environment you're comfortable
with, such as a Python IDE, Jupyter notebook, or a Python terminal. The easiest
option is to open this tutorial directly in colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/model-zoo/examples/blob/master/pytorch-quickstart/quickstart.ipynb)

Installation
------------

Install the Model Zoo client library via pip:
"""

#!pip install modelzoo-client[torch]

##############################################################################
#
# To deploy and use your own models, you'll need to create an account and
# configure an API key. You can do so from the command line:

#!modelzoo auth

##############################################################################
#
# Load
# ----
#
# First, we'll need a PyTorch model (`nn.Module`) to deploy. For the sake of
# this quickstart, we'll use the ``torchvision`` package to load a pretrained
# [Mobile Net V2 image classification
# model](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/).

import torch
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True)

##############################################################################
#
# Deploy
# ------
#
# To deploy this pipeline to a production-ready HTTP endpoint, use the
# [`modelzoo.torch.deploy()`](https://docs.modelzoo.dev/reference/modelzoo.torch.html#modelzoo.torch.deploy)
# function. Since the Model Zoo PyTorch client library relies on the ONNX
# format for serialization. `modelzoo.torch.deploy()` uses the same arguments
# as
# [`torch.onnx.export()`](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export),
# except that it does not require a filename. Instead, the ONNX-serialized
# model will be directly uploaded to Model Zoo.

import modelzoo.torch

model_name = modelzoo.torch.deploy(
    model, torch.randn(1, 3, 224, 224), input_names=["input"], output_names=["output"]
)

##############################################################################
#
# That's all there is to it! Behind the scenes, Model Zoo serialized your model
# to ONNX, uploaded it to object storage, deployed a container to serve any
# HTTP requests made to the model, and set up a load balancer to route requests
# to multiple model shards. If you'd like, take some time to explore the model
# via the Web UI link. There you'll be able to modify documentation, test the
# model with raw or visual inputs, monitor metrics and/or logs. By default,
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
# First, let's load a picture of a dog to use as a test for the image
# classification model. We'll also load the JSON metadata that maps the class
# indices to human-readable labels.

from PIL import Image
from io import BytesIO
import requests

image_response = requests.get(
    "https://modelzoo-datasets.s3-us-west-2.amazonaws.com/imagenet/dog.jpg"
)
input_image = Image.open(BytesIO(image_response.content))
input_image.show()

class_to_labels = requests.get(
    "https://modelzoo-datasets.s3-us-west-2.amazonaws.com/imagenet/class_idx_to_labels.json"
).json()

##############################################################################
#
# Next, we'll use our Python client library to query the model for a
# prediction.
# [`modelzoo.torch.predict()`](https://docs.modelzoo.dev/reference/modelzoo.torch.html#modelzoo.torch.predict)
# requires the `model_name` and a Python dictionary mapping input layer string
# names to a dictionary in
# [onnx.TensorProto](https://github.com/onnx/onnx/blob/9b7c2b4f0b4a16a0cf31145eae9425abe7cbe2a9/onnx/onnx-ml.proto#L451)
# format. In this example, we'll also transform the input image and apply the
# appropriate transformations as specified by the [model
# documentation](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/).

import modelzoo.torch
import numpy as np

import onnx
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)

full_output = modelzoo.torch.predict(
    model_name,
    {
        "input": {
            "dims": input_tensor.shape,
            "data_type": onnx.TensorProto.DataType.FLOAT,
            "float_data": input_tensor.numpy().tolist(),
        }
    },
)

output = full_output["outputs"]["output"]["floatData"]
model_prediction_idx = str(np.argmax(output))
print("Model prediction: {}".format(class_to_labels[model_prediction_idx]))

##############################################################################
#
# Great! At this point, we've used our image classification model to
# successfully make a prediction on a new image.
#
# Manage
# ------
#
# By default, Model Zoo will deploy your model and wait for it to get into a
# `HEALTHY` state, meaning that it's ready for predictions. You can always
# check on the state of a model by using the
# [`modelzoo.info()`](https://docs.modelzoo.dev/reference/modelzoo.html#modelzoo.info)
# function:

modelzoo.info(model_name)

##############################################################################
#
# To save resources and shut down any model if you aren't using it, you can use
# [`modelzoo.stop()`](https://docs.modelzoo.dev/reference/modelzoo.html#modelzoo.stop):

modelzoo.stop(model_name)

##############################################################################
#
# With Model Zoo you can manage model state manually, or automatically. By
# default, our free trial will stop any model where there has been no request
# activity for 15 minutes, saving you resources if you forget to stop manually.
# Our unlimited version has more options for controlling autoscaling behavior.
#
# Interested in what you've seen and want to test drive an unlimited version of
# Model Zoo? Apply to our [private
# beta](https://modelzoo.typeform.com/to/Y8U9Lw) and reach out at
# [contact@modelzoo.dev](mailto:contact@modelzoo.dev) to learn more.
