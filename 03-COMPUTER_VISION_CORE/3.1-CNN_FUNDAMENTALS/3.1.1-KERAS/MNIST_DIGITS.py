import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print("TF version: ", tf.__version__)

##DATASET
dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

##LAYERS
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)


predictions = model(x_train[:1]).numpy()
softmax = tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=20)
model.evaluate(x_test, y_test, verbose=3)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print(probability_model(x_test[:5]))

output = probability_model.predict(x_test[:5])

for i in range(5):
    plt.imshow(x_test[i])
    predicted_label = np.argmax(output[i])
    actual_label = y_test[i]
    plt.title(f"Model guessed: {predicted_label} | Actual: {actual_label}")
    plt.show()
