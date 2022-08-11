import glob
import os
from typing import Tuple

import h5py
import napari
import numpy as np
from scipy import ndimage as ndi

# The path to the folder containing the data
DIRECTORY_PATH = "/local0/kevin/napari-follicle-labeler/examples/test_files/"


# dataset key for the raw image
RAW_KEY = "raw_rescaled"

# dataset key for the follicles
FOLLICLE_KEY = "follicle_labels_rescaled"

# viewer setup constants
RAW_IMAGE_LAYER_NAME = "raw"
FOLLICLE_IMAGE_LAYER_NAME = "follicles"


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the raw image and follicles from a specified dataset.

    Parameters
    ----------
    dataset_path : str
        path to the dataset

    Returns
    -------
    raw_image : np.ndarray
        The raw image.
    follicle_labels : np.ndarray
        The follicle label image
    """
    with h5py.File(dataset_path, "r") as f:
        raw_image = f[RAW_KEY][:]
        follicle_mask = f[FOLLICLE_KEY][:]
    follicle_labels, _ = ndi.label(follicle_mask)
    return raw_image, follicle_labels


def _add_dataset_to_viewer(
    viewer: napari.Viewer,
    raw_image: np.ndarray,
    follicle_labels: np.ndarray,
    image_path: str,
    raw_image_layer_name: str = "raw",
    follicle_labels_layer_name: str = "follicles",
) -> None:
    image_layer = viewer.layers[raw_image_layer_name]
    image_layer.data = raw_image
    image_layer.metadata["image_path"] = image_path
    viewer.layers[follicle_labels_layer_name].data = follicle_labels


def _initialize_dataset(
    viewer: napari.Viewer, follicle_labels_layer_name: str = "follicles"
) -> None:
    labels_layer = viewer.layers[follicle_labels_layer_name]

    # get the non-background label values
    label_indices = np.unique(labels_layer.data)
    label_values = label_indices[label_indices != 0]

    # reset the current label index
    labels_layer.metadata["current_label_index"] = 0
    labels_layer.metadata["label_values"] = label_values

    # hack to force napari to render all colors
    # otherwise bug when displaying only one label
    viewer.dims.ndisplay = 3
    viewer.dims.ndisplay = 2

    # view only currently selected label\
    labels_layer.show_selected_label = True
    labels_layer.selected_label = label_values[0]


def _setup_viewer(initial_dataset_path: str) -> napari.Viewer:

    initial_raw_image, initial_follicle_labels = load_dataset(
        initial_dataset_path
    )

    viewer = napari.Viewer()
    viewer.add_image(
        initial_raw_image,
        metadata={"image_path": initial_dataset_path},
        name="raw",
    )
    viewer.add_labels(initial_follicle_labels, name="follicles")

    _initialize_dataset(viewer)

    return viewer


file_pattern = os.path.join(DIRECTORY_PATH, "*.h5")
dataset_file_paths = glob.glob(file_pattern)

current_dataset_index = 0
initial_file_path = dataset_file_paths[current_dataset_index]
viewer = _setup_viewer(initial_file_path)


if __name__ == "__main__":
    napari.run()
