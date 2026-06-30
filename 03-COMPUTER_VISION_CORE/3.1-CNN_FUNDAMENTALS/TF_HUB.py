import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

print(tf.config.list_physical_devices())

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=("train[:60%]", "train[60%:]", "test"),
    as_supervised=True,
)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch, train_labels_batch)


embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string),
        hub_layer,
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(
    train_data.shuffle(1000).batch(512),
    epochs=10,
    validation_data=validation_data.batch(512),
    verbose=1,
)

results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
