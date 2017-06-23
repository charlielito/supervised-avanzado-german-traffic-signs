from dataget import data # <== dataget
import tensorflow as tf
import cytoolz as cz
from phi.api import *
from model import Model
import numpy as np
import random
from name import network_name, model_path
from tfinterface.supervised import SupervisedInputs

# seed: resultados repetibles
seed = 32
np.random.seed(seed=seed)
random.seed(seed)

# dataget
dataset = data("german-traffic-signs").get()

# obtener todas las imagenes (lento)
data_generator = dataset.training_set.random_batch_arrays_generator(64)
data_generator = cz.map(Dict(features = P[0], labels = P[1]), data_generator)

graph = tf.Graph()
sess = tf.Session(graph=graph)

# inputs
inputs = SupervisedInputs(
    name = network_name + "_inputs",
    graph = graph,
    sess = sess,
    # tensors
    features = dict(shape = (None, 32, 32, 3)),
    labels = dict(shape = (None,), dtype = tf.uint8)
)


# create model template
template = Model(
    n_classes = 43,
    name = network_name,
    model_path = model_path,
    graph = graph,
    sess = sess,
    seed = seed,
    optimizer = tf.train.AdamOptimizer,
)

# model
assert template.graph is inputs.graph

inputs = inputs()
model = template(inputs)

# initialize variables
model.initialize()

# fit
print("training")
model.fit(
    data_generator = data_generator,
    epochs = 6000,
    log_summaries = True,
    log_interval = 10,
    print_test_info = True,
)

# save
print("saving model")
model.save()
