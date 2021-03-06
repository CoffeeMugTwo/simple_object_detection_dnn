import numpy as np
import luigi
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from src.data.preprocess_dataset import PreprocessData


class TrainOneRectangleDNN(luigi.Task):

    _path_to_data = Path("data/raw/image_array.npy")
    _path_to_labels = Path("data/raw/bbox_array.npy")
    _path_to_model = Path("models/single_rec_dnn")
    _path_to_train_hist_plot = Path("reports/figures/single_rec_dnn.pdf")

    def requires(self):
        yield PreprocessData()

    def output(self):
        return [luigi.LocalTarget(self._path_to_model),
                luigi.LocalTarget(self._path_to_train_hist_plot)]

    def run(self):

        # load raw data
        img_array = np.load((self.input()[0][0]).path)
        labels = np.load((self.input()[0][1]).path)

        # splint into train and test data
        x_train, x_test, y_train, y_test = train_test_split(img_array,
                                                            labels,
                                                            test_size=0.2)

        # define model
        model = keras.Sequential([
            layers.Dense(200, activation="relu", input_dim=64),
            layers.Dropout(0.2),
            layers.Dense(4)
        ])

        model.compile(optimizer="adadelta",
                      loss='mse')

        # train model
        training_history = model.fit(x_train,
                                     y_train,
                                     epochs=256,
                                     validation_data=(x_test, y_test))

        # plot training
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(training_history.history['loss'],
                label='Training Loss',
                marker=".")
        ax.plot(training_history.history['val_loss'],
                label='Validation Loss',
                marker='.')
        ax.set_xlabel("Epochs")
        ax.grid(True)

        fig.legend()

        fig.savefig(self._path_to_train_hist_plot)

        print(training_history.history)

        # store model
        model.save(self._path_to_model)

        return


if __name__ == "__main__":
    luigi.build([TrainOneRectangleDNN()],
                local_scheduler=True)