"""app to view and curate labels

1. Update the DIRECTORY_PATH, ANNOTATION_DIRECTORY, and OUTPUT_DIRECTORY
2. enter and image file name in: file_name
3. Modify the labels in the "annotations" layer
4. When done, press "s" to save the updated labels to the OUTPUT_DIRECTORY

"""

import glob
import os
import warnings
from typing import Dict, List, Tuple

import h5py
import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from scipy import ndimage as ndi

# import numpy as np
# from scipy import ndimage as ndi
# len(np.unique(ndi.label(viewer.layers["annotations"].data)[0]))
# == len(np.unique(ndi.label(viewer.layers["follicles"].data)[0]))


# The path to the folder containing the data
DIRECTORY_PATH = "./test_data/"

# get all of the files
file_pattern = os.path.join(DIRECTORY_PATH, "*.h5")
dataset_file_paths = sorted(glob.glob(file_pattern))

# initialize the current dataset index
# used to determine which file path to load
current_dataset_index = 0  # finished dataset_index: 23

# file_name = "A01_z1_links_170423_ovary.h5"

# The path to the directory in which the follicle labels are saved
# (CSV from follicle_labeler)
ANNOTATION_DIRECTORY = "./annotations/"

# the path to the directory in which the updated labels will be saved
OUTPUT_DIRECTORY = "./curated_annotations/"

# dataset key for the raw image
RAW_KEY = "raw_rescaled"

# dataset key for the follicles
FOLLICLE_KEY = "follicle_labels_rescaled"

# color map for the annotated follicles (RGBA)
ANNOTATION_COLORMAP = {
    "background": [0, 0, 0, 0],
    "good": [127, 201, 127, 255],
    "merge": [190, 174, 212, 255],
    "over": [253, 192, 134, 255],
    "under": [255, 255, 153, 255],
    "split": [56, 108, 176, 255],
    "false_positive": [240, 2, 127, 255],
    "unannotated": [0, 0, 0, 255],
}


if not os.path.isdir(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)


def load_dataset(
    dataset_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw image, follicle label and follicle annotation from
    a specified dataset. If a curated annotation file exists, load that
    for that annotation instead of copying the label.

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
    curated_annotation : np.ndarray
        The annotation image
    """
    with h5py.File(dataset_path, "r") as f:
        raw_image = f[RAW_KEY][:]
        follicle_mask = f[FOLLICLE_KEY][:]
    follicle_labels, _ = ndi.label(follicle_mask)

    file_name = os.path.basename(dataset_path)

    # get the curated annotation file path
    output_file_path = os.path.join(OUTPUT_DIRECTORY, file_name)

    # load existing curated annotation if it exist
    if os.path.isfile(output_file_path):
        with h5py.File(output_file_path, "r") as f:
            curated_annotation = f["curated_follicle_labels"][:]
        warnings.warn(f"Loading curated annotation from {output_file_path}")
    else:
        curated_annotation = follicle_labels.copy()
        warnings.warn(f"No curated annotation found for {output_file_path}")

    return raw_image, follicle_labels, curated_annotation


def _increment_index(current_index: int, n_indices: int):
    """Increment an index with wraparound."""
    return (current_index + 1) % n_indices


def _decrement_index(current_index: int, n_indices: int):
    """Decrement an index with wraparound."""
    return ((current_index - 1) + n_indices) % n_indices


def _initialize_dataset(
    viewer: napari.Viewer,
    raw_image_layer_name: str = "raw",
    follicle_labels_layer_name: str = "follicles",
) -> None:
    labels_layer = viewer.layers[follicle_labels_layer_name]

    # get the non-background label values
    label_indices = np.unique(labels_layer.data)

    # hack to force napari to render all colors
    # otherwise bug when displaying only one label
    viewer.dims.ndisplay = 3
    viewer.dims.ndisplay = 2

    # show only the outline of the follicles
    labels_layer.contour = 2

    # initialize the table
    image_file_path = viewer.layers[raw_image_layer_name].metadata[
        "image_path"
    ]
    image_file_name = os.path.basename(image_file_path)
    table_file_path = os.path.join(
        ANNOTATION_DIRECTORY, image_file_name.replace(".h5", ".csv")
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

    # set the colormap
    annotations_layer = viewer.layers["annotations"]

    colormap = dict()
    for i, row in follicle_table.iterrows():
        annotation_value = row["annotation"]
        color = ANNOTATION_COLORMAP.get(
            annotation_value, ANNOTATION_COLORMAP["unannotated"]
        )
        colormap[row["index"]] = np.asarray(color) / 255
    annotations_layer.color = colormap
    annotations_layer.color_mode = "direct"

    merged_follicles = follicle_table.index[
        follicle_table["annotation"].str.contains("merge")
    ].to_list()
    annotations_layer.selected_label = (
        merged_follicles[0] if len(merged_follicles) > 0 else 1
    )
    fp_follicles = follicle_table.index[
        follicle_table["annotation"].str.contains("false_positive")
    ].to_list()
    warnings.warn(
        f"Loaded image: {image_file_path}. Follicles {merged_follicles} "
        f"are merges. Follicles {fp_follicles} are merges."
    )


def _setup_viewer(initial_dataset_path: str) -> napari.Viewer:
    """Create the initial viewer object"""

    # load the first dataset
    (
        initial_raw_image,
        initial_follicle_labels,
        curated_annotation,
    ) = load_dataset(initial_dataset_path)

    # create the viewer
    viewer = napari.Viewer()

    _add_dataset_to_viewer(
        viewer,
        raw_image=initial_raw_image,
        follicle_labels=initial_follicle_labels,
        curated_annotations=curated_annotation,
        image_path=initial_dataset_path,
    )

    # initialize the viewer to annotate the current dataset
    _initialize_dataset(viewer)

    return viewer


def _add_dataset_to_viewer(
    viewer: napari.Viewer,
    raw_image: np.ndarray,
    follicle_labels: np.ndarray,
    curated_annotations: np.ndarray,
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
    viewer.add_labels(follicle_labels, name="follicles", visible=False)
    viewer.add_labels(curated_annotations, name="annotations")


def _load_next_file(viewer: napari.Viewer, direction="forward") -> None:

    # get the current index and paths
    # this will be a class property when moved into plugin
    global dataset_file_paths
    global current_dataset_index

    print(f"finished dataset_index: {current_dataset_index}")

    # get the next file name
    n_files = len(dataset_file_paths)
    if direction == "forward":
        current_dataset_index = _increment_index(
            current_dataset_index, n_files
        )
    elif direction == "backward":
        current_dataset_index = _decrement_index(
            current_dataset_index, n_files
        )
    new_dataset_path = dataset_file_paths[current_dataset_index]

    # load the data
    raw_image, follicle_labels, curated_annotations = load_dataset(
        new_dataset_path
    )

    # add the new data to the viewer
    _add_dataset_to_viewer(
        viewer,
        raw_image=raw_image,
        follicle_labels=follicle_labels,
        curated_annotations=curated_annotations,
        image_path=new_dataset_path,
    )

    # initialize the new layers
    _initialize_dataset(viewer)


def save_labels(viewer: napari.Viewer) -> None:
    # get the updated annotations and labels
    curated_labels = viewer.layers["annotations"].data
    labels = viewer.layers["follicles"].data

    # Check if there are any changes to the annotations
    if not np.array_equal(curated_labels, labels):
        # get the file name
        file_path = viewer.layers["raw"].metadata["image_path"]
        file_name = os.path.basename(file_path)

        # get the new file path
        output_file_path = os.path.join(OUTPUT_DIRECTORY, file_name)

        # save the file
        with h5py.File(output_file_path, "w") as f_out:
            f_out.create_dataset(
                name="curated_follicle_labels",
                data=curated_labels,
                compression="gzip",
            )

        warnings.warn(f"Saved labels to: {output_file_path}")
    else:
        warnings.warn(
            "Nothing to save: No changes were made to the annotation layer."
        )


def save_labels_and_forward_file(viewer: napari.Viewer) -> None:
    save_labels(viewer)
    _load_next_file(viewer, direction="forward")


def save_labels_and_backward_file(viewer: napari.Viewer) -> None:
    save_labels(viewer)
    _load_next_file(viewer, direction="backward")


class ColorLegend(QWidget):
    def __init__(self, colormap: Dict[str, List[int]]):
        super().__init__()
        self.colormap = colormap

        self.setLayout(QVBoxLayout())

        for label, color in self.colormap.items():
            if label == "background":
                continue
            label_widget = QLabel(label)
            if label == "unannotated":
                font_color = "white"
            else:
                font_color = "black"
            color_string = (
                f"color: {font_color};"
                f"background-color: rgb({color[0]},{color[1]},{color[2]})"
            )
            label_widget.setStyleSheet(color_string)
            self.layout().addWidget(label_widget)


initial_file_path = dataset_file_paths[current_dataset_index]
print(f"initial_file_path: {initial_file_path}")

viewer = _setup_viewer(initial_file_path)

# add the widget to show the colors
color_legend_widget = ColorLegend(ANNOTATION_COLORMAP)
viewer.window.add_dock_widget(color_legend_widget)

# add the hotkey for saving the labels
viewer.bind_key("s", save_labels_and_forward_file)
viewer.bind_key("a", save_labels_and_backward_file)


if __name__ == "__main__":
    napari.run()
