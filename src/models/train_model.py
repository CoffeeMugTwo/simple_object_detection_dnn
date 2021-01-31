import numpy as np
import luigi

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TrainOneRectangleDNN(luigi.Task):

    _

    def run(self):

        model = keras.Sequential([
            layers.Dense(256, activation="relu", input_dim=64),
            layers.Dropout(0.2)
            layers.Dense(256, activation="relu")
            layers.Dropout(0.2)
            layers.Dense(4)
        ])

        model.compile(optimizer="adadelta",
                      loss='mse')

