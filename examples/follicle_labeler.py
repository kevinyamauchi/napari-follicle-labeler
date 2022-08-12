"""prototype proofing GUI for follicle labels.

Annotations are stored in CSV files (one file per image).
Follicles can be annotated as:
     "good" (follicle is good)
     "merge" (merged follicles)
     "over" (annotation is too big)
     "under" (annotation is too small)
When you re-open an image, the saved annotations are automatically loaded.

Instructions:
1. update the DIRECTORY_PATH to the path containing your datasets
2. update the OUTPUT_DIRECTORY to the path where you want your
   annotations to be saved
3. run the script
4. the first image will be loaded. one follicle will be shown at a time.
   Use the hot keys to annotate. when you press the key, the next follicle
   to be annotated will be automatically shown.
     q - "good" (follicle is good)
     w - "merge" (merged follicles)
     e - "over" (annotation is too big)
     r - "under" (annotation is too small)
   When you annotate a follicle, your action is confirmed and the
   fraction of follicles you have annotated is displayed in a pop up
   and in the terminal.

   You can also manually move forward and backwards:
     d - show previous follicle
     f - show next follicle
   You can save your current annotations (will overwrite existing one)
     s - save current annotations (only for the open file)
   When you are done annotating a file, you can save and load the next file.
   You will see a message printed in your terminal
   if there are unannotated follicles remaining.
"""

import glob
import os
import time
import warnings
from typing import Optional, Tuple

import h5py
import napari
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# The path to the folder containing the data
DIRECTORY_PATH = "./test_data"


# The path to the directory in which the annotations will be saved
OUTPUT_DIRECTORY = "./annotations"

if not os.path.isdir(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)


# dataset key for the raw image
RAW_KEY = "raw_rescaled"

# dataset key for the follicles
FOLLICLE_KEY = "follicle_labels_rescaled"

# viewer setup constants
RAW_IMAGE_LAYER_NAME = "raw"
FOLLICLE_IMAGE_LAYER_NAME = "follicles"

# label values for annotating follicles
# todo: make enum
GOOD_ANNOTATION = "good"

# multiple follicles merged together
MERGE_ANNOTATION = "merge"

# follicle is over painted (too big)
OVER_ANNOTATION = "over"

# follicle is under painted (too small)
UNDER_ANNOTATION = "under"

# initialize the current dataset index
# used to determine which file path to load
current_dataset_index = 0

# get all of the files
file_pattern = os.path.join(DIRECTORY_PATH, "*.h5")
dataset_file_paths = sorted(glob.glob(file_pattern))


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
    # image_layer = viewer.layers[raw_image_layer_name]
    # image_layer.data = raw_image
    # image_layer.metadata["image_path"] = image_path
    # viewer.layers[follicle_labels_layer_name].data = follicle_labels

    viewer.layers.clear()

    viewer.add_image(
        raw_image,
        metadata={"image_path": image_path},
        name="raw",
    )
    viewer.add_labels(follicle_labels, name="follicles")


def _initialize_dataset(
    viewer: napari.Viewer,
    raw_image_layer_name: str = "raw",
    follicle_labels_layer_name: str = "follicles",
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

    # view only currently selected label
    labels_layer.show_selected_label = True
    labels_layer.selected_label = label_values[0]

    # show only the outline of the follicles
    labels_layer.contour = 2

    # initialize the table
    image_file_path = viewer.layers[raw_image_layer_name].metadata[
        "image_path"
    ]
    image_file_name = os.path.basename(image_file_path)
    table_file_path = os.path.join(
        OUTPUT_DIRECTORY, image_file_name.replace(".h5", ".csv")
    )
    if os.path.isfile(table_file_path):
        follicle_table = pd.read_csv(table_file_path)
    else:
        file_name = os.path.basename(table_file_path)
        n_indices = len(label_indices)
        empty_annotations = np.empty((n_indices,))
        empty_annotations[:] = np.nan
        follicle_table = pd.DataFrame(
            {
                "file_path": [file_name] * n_indices,
                "index": label_indices,
                "annotation": empty_annotations,
            }
        )
        follicle_table.set_index("index", drop=False, inplace=True)

        # set the annotation for the background label (0)
        follicle_table.at[0, "annotation"] = "background"
    labels_layer.metadata["annotations"] = follicle_table


def _setup_viewer(initial_dataset_path: str) -> napari.Viewer:
    """Create the initial viewer object"""

    # load the first dataset
    initial_raw_image, initial_follicle_labels = load_dataset(
        initial_dataset_path
    )

    # create the viewer
    viewer = napari.Viewer()
    viewer.add_image(
        initial_raw_image,
        metadata={"image_path": initial_dataset_path},
        name="raw",
    )
    viewer.add_labels(initial_follicle_labels, name="follicles")

    # initialize the viewer to annotate the current dataset
    _initialize_dataset(viewer)

    return viewer


def _increment_index(current_index: int, n_indices: int):
    """Increment an index with wraparound."""
    return (current_index + 1) % n_indices


def _decrement_index(current_index: int, n_indices: int):
    """Decrement an index with wraparound."""
    return ((current_index - 1) + n_indices) % n_indices


def _select_previous_label(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:

    # get the current state
    labels_layer = viewer.layers[FOLLICLE_IMAGE_LAYER_NAME]
    current_label_index = labels_layer.metadata["current_label_index"]
    labels_values = labels_layer.metadata["label_values"]
    n_label_values = len(labels_values)

    # decrement the label index and get the new value
    new_label_index = _decrement_index(current_label_index, n_label_values)
    new_label_value = labels_values[new_label_index]

    # set the new layer state
    labels_layer.metadata["current_label_index"] = new_label_index
    labels_layer.selected_label = new_label_value


def _select_next_label(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:

    # get the current state
    labels_layer = viewer.layers[FOLLICLE_IMAGE_LAYER_NAME]
    current_label_index = labels_layer.metadata["current_label_index"]
    labels_values = labels_layer.metadata["label_values"]
    n_label_values = len(labels_values)

    # decrement the label index and get the new value
    new_label_index = _increment_index(current_label_index, n_label_values)
    new_label_value = labels_values[new_label_index]

    # set the new layer state
    labels_layer.metadata["current_label_index"] = new_label_index
    labels_layer.selected_label = new_label_value


def _annotate_selected_label(
    viewer: napari.Viewer, annotation: str, go_to_next_label: bool = True
) -> None:
    labels_layer = viewer.layers[FOLLICLE_IMAGE_LAYER_NAME]
    follicle_table = labels_layer.metadata["annotations"]

    # get the currently selected label and annotate
    label_value = labels_layer.selected_label
    follicle_table.at[label_value, "annotation"] = annotation

    # get the number of annotated_follicles
    # -1 because background is included
    n_annotated_follicles = follicle_table["annotation"].notnull().sum() - 1

    # get the total number of follicles
    # -1 because background is included
    n_follicles = len(follicle_table) - 1

    confirmation_message = (
        f"follicle {label_value} is {annotation}. "
        + f"{n_annotated_follicles}/{n_follicles} follicles annotated"
    )
    warnings.warn(confirmation_message)

    if go_to_next_label is True:
        _select_next_label(viewer)


def annotate_good_follicle(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:
    """annotate the selected follicle as good"""
    _annotate_selected_label(viewer, GOOD_ANNOTATION, go_to_next_label=True)


def annotate_merge_follicle(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:
    """annotate the selected follicle as merged"""
    _annotate_selected_label(viewer, MERGE_ANNOTATION, go_to_next_label=True)


def annotate_over_follicle(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:
    """annotate the selected follicle as over labeled (too big)"""
    _annotate_selected_label(viewer, OVER_ANNOTATION, go_to_next_label=True)


def annotate_under_follicle(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:
    """annotate the selected follicle as under labeled (too small)"""
    _annotate_selected_label(viewer, UNDER_ANNOTATION, go_to_next_label=True)


def save_annotations(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:
    """Save the current annotations.

    Annotations are saved as CSV the OUTPUT_DIRECTORY with the same name as the
    image file name
    """
    # get the annotations
    follicle_table = viewer.layers[FOLLICLE_IMAGE_LAYER_NAME].metadata[
        "annotations"
    ]

    # get the table file path
    image_file_path = viewer.layers[RAW_IMAGE_LAYER_NAME].metadata[
        "image_path"
    ]
    image_file_name = os.path.basename(image_file_path)
    table_file_path = os.path.join(
        OUTPUT_DIRECTORY, image_file_name.replace(".h5", ".csv")
    )

    follicle_table.to_csv(table_file_path)
    warnings.warn(f"annotations saved to: {table_file_path}")


def _load_next_file(viewer: napari.Viewer) -> None:

    # get the current index and paths
    # this will be a class property when moved into plugin
    global dataset_file_paths
    global current_dataset_index

    # get the next file name
    n_files = len(dataset_file_paths)
    current_dataset_index = _increment_index(current_dataset_index, n_files)
    new_dataset_path = dataset_file_paths[current_dataset_index]

    # load the data
    raw_image, follicle_labels = load_dataset(new_dataset_path)

    # add the new data to the viewer
    _add_dataset_to_viewer(
        viewer,
        raw_image=raw_image,
        follicle_labels=follicle_labels,
        image_path=new_dataset_path,
    )

    # initialize the new layers
    _initialize_dataset(viewer)


def save_and_next_file(
    viewer: Optional[napari.Viewer] = None, event=None
) -> None:
    # save the annotations
    save_annotations(viewer)

    # check that all annotations were completed
    follicle_table = viewer.layers[FOLLICLE_IMAGE_LAYER_NAME].metadata[
        "annotations"
    ]
    unannotated_follicle_table = follicle_table.loc[
        follicle_table["annotation"].isna()
    ]

    if len(unannotated_follicle_table) > 0:
        unannotated_follicles = unannotated_follicle_table["index"].to_numpy()
        warnings.warn(f"{unannotated_follicles} have not been annotated")
        time.sleep(1)

    # advance to the next file
    _load_next_file(viewer)


initial_file_path = dataset_file_paths[current_dataset_index]
viewer = _setup_viewer(initial_file_path)

# add key bindings to increment/decrement labels
viewer.bind_key("d", _select_previous_label)
viewer.bind_key("f", _select_next_label)

# add key bindings to annotate selected follicle
viewer.bind_key("q", annotate_good_follicle)
viewer.bind_key("w", annotate_merge_follicle)
viewer.bind_key("e", annotate_over_follicle)
viewer.bind_key("r", annotate_under_follicle)

# add key binding to save annotations
viewer.bind_key("s", save_annotations)

# add key binding to go to the next file
viewer.bind_key("a", save_and_next_file)


if __name__ == "__main__":
    napari.run()
