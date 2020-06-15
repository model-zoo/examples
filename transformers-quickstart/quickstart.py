"""
Transformers: Quick Start
=========================

In this tutorial, we are going to deploy a language model to Model Zoo with
[HuggingFace Transformers](https://huggingface.co/transformers/) and use it to
generate an original passage of text.

You can follow along this tutorial in any Python environment you're comfortable
with, such as a Python IDE, Jupyter notebook, or a Python terminal. The easiest
option is to open this tutorial directly in colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/model-zoo/examples/blob/master/transformers-quickstart/quickstart.ipynb)

Installation
------------

Install the Model Zoo client library via pip:
"""

#!pip install modelzoo-client

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
# First, we'll need to train or load a model and tokenizer in the form of a
# [transformers.Pipeline](https://huggingface.co/transformers/main_classes/pipelines.html)
# object. Pipeline offers a simple API dedicated to several tasks, including
# Named Entity Recognition, Masked Language Modeling, Sentiment Analysis,
# Feature Extraction and Question Answering. Model Zoo relies on this API to
# figure out how to package and serve your model behind an HTTP endpoint.
#
# For the sake of this quickstart, we'll use the default "text-generation"
# pipeline which loads the [OpenAI
# GPT-2](https://openai.com/blog/better-language-models/) model and tokenizer.

import transformers

pipeline = transformers.pipeline("text-generation")

##############################################################################
#
# The Model Zoo free tier currently supports
# [`transformers.TextGenerationPipeline`](https://huggingface.co/transformers/main_classes/pipelines.html?highlight=transformers.TextGenerationPipeline#transformers.TextGenerationPipeline).
# If you're interested in deploying other types of pipelines, please reach out
# to [contact@modelzoo.dev](mailto:contact@modelzoo.dev).
#
# Deploy
# ------
#
# To deploy this pipeline to a production-ready HTTP endpoint, use the
# [`modelzoo.transformers.deploy()`](https://docs.modelzoo.dev/reference/modelzoo.transformers.html#modelzoo.transformers.deploy)
# function. Since GPT2 is a large model with high memory requirements, we
# override defaults to configure our containers to use 2 GB memory and 1024 CPU
# units (1 vCPU) with
# [`modelzoo.ResourcesConfig`](https://docs.modelzoo.dev/reference/modelzoo.html#modelzoo.ResourcesConfig).

import modelzoo.transformers

model_name = modelzoo.transformers.deploy(
    pipeline=pipeline,
    resources_config=modelzoo.ResourcesConfig(memory_mb=2048, cpu_units=1024),
)

##############################################################################
#
# That's all there is to it! Behind the scenes, Model Zoo serialized your
# pipeline, deployed it to a container to serve HTTP requests, and set up a
# load balancer to route requests to multiple model shards.  If you'd like,
# take some time to explore the model via the Web UI link. There you'll be able
# to modify documentation, test the model with raw or visual inputs, monitor
# metrics and/or logs.
#
# You can specify the name of the model you'd like to deploy via a ``model_name``
# argument. If a name is omitted, Model Zoo will choose a unique one for you.
# Model names must be unique to your account.
#
# Predict
# -------
#
# Now that the model is deployed, you can use our Python client API to query
# the language model to generate some text.
# [`modelzoo.transformers.generate()`](https://docs.modelzoo.dev/reference/modelzoo.transformers.html#modelzoo.transformers.generate)
# function requires the `model_name` and an optional input string to prime the
# text generation pipeline.

print(
    modelzoo.transformers.generate(
        model_name, input_str="These violent delights have violent ends"
    )
)

##############################################################################
#
# Great! At this point, we've used our language model to generate an original
# passage of text.
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
