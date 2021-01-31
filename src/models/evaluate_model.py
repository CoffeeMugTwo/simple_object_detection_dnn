import numpy as np
import luigi
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from src.data.make_dataset import MakeDataset
from src.models.train_model import TrainOneRectangleDNN

class EvaluateOneRectangleDNN(luigi.Task):

    _n_plots = 10
    _path_to_test_data = Path("data/raw/")
    _path_to_model = Path("models/single_rec_dnn")
    _path_to_figures = Path("reports/figures/")
    _path_to_eval_results = Path("reports")

    def requires(self):
        yield MakeDataset()
        yield TrainOneRectangleDNN()

    def run(self):
        # load test data
        x_test = np.load(self._path_to_test_data / f"eval_images_array.npy")
        y_test = np.load(self._path_to_test_data / f"eval_bbox_array.npy")

        # load model to evaluate
        model = tf.keras.models.load_model(self._path_to_model)

        # Use model to predict
        y_pred = model.predict(x_test)

        # draw rectangles, true bboxes and predicted bboxes for n_plot cases
        fig = plt.figure()
        for i_plot in range(self._n_plots):
            fig.clear()
            ax = fig.add_subplot(111)
            img = x_test[i_plot].reshape(8, 8).T
            ax.imshow(img,
                      origin="lower",
                      extent=(0, 8, 0, 8))
            print(y_pred[i_plot])
            print(y_test[i_plot])
            # add predicted bbox
            self.add_bbox(ax,
                          y_pred[i_plot],
                          color="r",
                          label="Predicted")
            # add true bbox
            self.add_bbox(ax,
                          y_test[i_plot][0],
                          color="b",
                          label="True")
            # store plot
            fig.savefig(self._path_to_figures / f"pred_fig_{i_plot}.pdf")

        # Calculate mean IOU
        iou = 123

        # Store mean iou
        np.savetxt(self._path_to_eval_results / "single_rec_dnn_iou.txt",
                   np.array([iou]))


    @staticmethod
    def add_bbox(ax, bbox, color="r", label=None, fill=False):
        """Adds a hollow bounding box to an axis object

        Parameters
        ----------
        ax : matplotlib axis object
        bbox : array
            Array containing the bounding box properties [x, y, w, h]
        color : str
            Color as string
        label : str
            Label for the respective bounding box
        """

        rect = patches.Rectangle(xy=(bbox[0], bbox[1]),
                                 width=bbox[2],
                                 height=bbox[3],
                                 fill=fill,
                                 color=color,
                                 label=label)

        ax.add_patch(rect)

        return

if __name__ == "__main__":
    luigi.build([EvaluateOneRectangleDNN()],
                local_scheduler=True)