import luigi
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm

class MakeDataset(luigi.Task):
    _data_folder_path = Path('data/raw/')

    _plot_images = False

    _num_imgs = 60000
    _img_size = 8
    _min_object_dim_size = 1
    _max_object_dim_size = 4
    _num_objects_per_img = 1


    def output(self):
        yield [luigi.LocalTarget(self._data_folder_path / f"image_array.npy"),
               luigi.LocalTarget(self._data_folder_path / f"bbox_array.npy"),
               luigi.LocalTarget(self._data_folder_path / f"eval_images_array.npy"),
               luigi.LocalTarget(self._data_folder_path / f"eval_bbox_array.npy")]

    def run(self):

        imges_arr = np.zeros((self._num_imgs, self._img_size, self._img_size)) # empty image arrays
        images_png = list() # empty list for images
        bboxes = np.zeros((self._num_imgs, self._num_objects_per_img, 4)) # dummy bounding boxes
        fig = plt.figure()


        for i_img in tqdm(range(self._num_imgs)):

            for i_object in range(self._num_objects_per_img):

                # get properties of rectangle
                width, height = np.random.randint(self._min_object_dim_size,
                                                  self._max_object_dim_size,
                                                  size=2) # width and height of rectangle
                x = np.random.randint(0, self._img_size - width) # lower left corner of rectangle
                y = np.random.randint(0, self._img_size - height) # upper left corner of rectangle

                # include rectangle to image
                imges_arr[i_img, x:x+width, y:y+height] = 1

                # store bbox of rectangle
                bboxes[i_img, i_object] = [x, y, width, height]

            if self._plot_images:
                # Create png from image array
                fig.clear()
                ax = fig.add_subplot(111)
                ax.imshow(imges_arr[i_img])
                ax.set_axis_off()
                fig.savefig(self._data_folder_path / f"i_img_{i_img}.png")

        # flatten image arrays
        flat_images = list()
        for i_img in tqdm(range(imges_arr.shape[0])):
            flat_img = imges_arr[i_img].flatten()
            flat_images.append(flat_img)

        # Store image array and bboxes
        # Set test data aside for evaluation
        x_train, x_test, y_train, y_test = train_test_split(flat_images,
                                                            bboxes,
                                                            test_size=0.1)
        np.save(self._data_folder_path / f"image_array.npy",
                np.array(x_train))
        np.save(self._data_folder_path / f"bbox_array.npy",
                y_train)
        np.save(self._data_folder_path / f"eval_images_array.npy",
                np.array(x_test))
        np.save(self._data_folder_path / f"eval_bbox_array.npy",
                y_test)

        return


if __name__ == "__main__":
    print("test")
    luigi.build([MakeDataset()],
                local_scheduler=True)


