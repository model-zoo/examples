TensorFlow: Quick Start
=======================

In this tutorial, we are going to deploy an image classifier to Model Zoo with
TensorFlow and use it to make sample predictions.

You can follow along this tutorial in any Python environment you're comfortable
with, such as a Python IDE, Jupyter notebook, or a Python terminal. The easiest
option is to open this tutorial directly in colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/model-zoo/examples/blob/master/tensorflow-quickstart/quickstart.ipynb)

Installation
------------

Install the Model Zoo client library via pip:

```python
!pip install modelzoo-client
```


To deploy and use your own models, you'll need to create an account and
configure an API key. You can do so from the command line:

```python
!modelzoo auth
```


Train
-----

First, we'll need a TensorFlow model to deploy. Model Zoo is solely focused on
deployment and monitoring of models, so you can feel free to train or load a
Tensorflow model using any method, tool, or infrastructure. For the purposes of
this demo, we'll use the TensorFlow official example to [classify images of
clothing](https://www.tensorflow.org/tutorials/keras/classification).

```python
import numpy as np
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

(
    (train_images, train_labels),
    (test_images, test_labels),
) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)
```


Deploy
------

To deploy this TensorFlow model to a production-ready HTTP endpoint, use the
[`modelzoo.tensorflow.deploy()`](https://docs.modelzoo.dev/reference/modelzoo.tensorflow.html#modelzoo.tensorflow.deploy)
function:

```python
import modelzoo.tensorflow

model_name = modelzoo.tensorflow.deploy(model)
```


That's all there is to it! Behind the scenes, Model Zoo serialized your model
to the TensorFlow SavedModel format, uploaded it to object storage, deployed
a container to serve any HTTP requests made to the model, and set up a load
balancer to route requests to multiple model shards. If you'd like, take some
time to explore the model via the Web UI link. There you'll be able to modify
documentation, test the model with raw or visual inputs, monitor metrics
and/or logs.

You can specify the name of the model you'd like to deploy via a ``model_name``
argument. If a name is omitted, Model Zoo will choose a unique one for you.
Model names must be unique to your account.

Predict
-------

Now that the model is deployed, you can use our Python client API to query
the model for a prediction. The
[`modelzoo.tensorflow.predict()`](https://docs.modelzoo.dev/reference/modelzoo.tensorflow.html#modelzoo.tensorflow.predict)
function requires the `model_name` and a payload for prediction -- in this
case a numpy array representing a test image.

```python
print("\nTest label ground truth:", test_labels[0])

scores = modelzoo.tensorflow.predict(model_name, test_images[0])
scores_by_class = list(zip(range(10), scores[0]))
model_prediction = np.argmax(scores)

print("\nModel scores by class:", scores_by_class)
print("\nModel prediction:", np.argmax(scores))
```


Great! At this point, we've successfully queried our deployed model for a
prediction on an image it hasn't seen during training.

Manage
------

By default, Model Zoo will deploy your model and wait for it to get into a
`HEALTHY` state, meaning that it's ready for predictions. You can always
check on the state of a model by using the
[`modelzoo.info()`](https://docs.modelzoo.dev/reference/modelzoo.html#modelzoo.info)
function:

```python
modelzoo.info(model_name)
```


To save resources and shut down any model if you aren't using it, you can use
[`modelzoo.stop()`](https://docs.modelzoo.dev/reference/modelzoo.html#modelzoo.stop):

```python
modelzoo.stop(model_name)
```


With Model Zoo you can manage model state manually, or automatically. By
default, our free trial will stop any model where there has been no request
activity for 15 minutes, saving you resources if you forget to stop manually.
Our unlimited version has more options for controlling autoscaling behavior.

Interested in what you've seen and want to test drive an unlimited version of
Model Zoo? Apply to our [private
beta](https://modelzoo.typeform.com/to/Y8U9Lw) and reach out at
[contact@modelzoo.dev](mailto:contact@modelzoo.dev) to learn more.
