import os
import h5py
import re
import numpy as np
from utils.ipa_utils import resize_squeeze


def convert_dir_to_hdf5(data_dir, output_dir, image_size=(480, 640, 3)):
    """
    Converts all subdirectories in `data_dir` to separate HDF5 files under `output_dir`.

    Args:
        data_dir (str): Path to the directory containing subdirectories with data.
        output_dir (str): Path to the output directory for HDF5 files.
        image_size (tuple, optional): Size of the images (height, width) in pixels. Defaults to (480, 640).
    """

    for subdir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, subdir)):
            continue

        scene_dir = os.path.join(data_dir, subdir)
        hdf5_path = os.path.join(output_dir, f"{subdir}.hdf5")

        with h5py.File(hdf5_path, "w") as hdf5_file:
            # Collect and validate file paths
            color_0_paths, color_1_paths, depth_0_paths, segmap_paths = collect_file_paths(scene_dir)

            # Create datasets in HDF5 file
            create_datasets(hdf5_file, color_0_paths, depth_0_paths, segmap_paths, image_size)

            # Load and store data in datasets
            for i, (
                color_0_path,
                color_1_path,
                depth_0_path,
                segmap_path,
            ) in enumerate(
                zip(
                    color_0_paths,
                    color_1_paths,
                    depth_0_paths,
                    segmap_paths,
                )
            ):
                load_and_store_data(
                    hdf5_file, color_0_path, color_1_path, depth_0_path, segmap_path, i, image_size
                )

        print(f"Saved data from {scene_dir} to {hdf5_path}")


def collect_file_paths(scene_dir):
    """
    Collects and validates file paths within a subdirectory.

    Args:
        scene_dir (str): Path to the subdirectory containing data files.


    Returns:
        tuple: A tuple containing lists of file paths for each data modality.
    """

    num_images = sum(
        1 for filename in os.listdir(scene_dir) if re.match(r".*_colors_0.png$", filename)
    )  # Assuming equal number of color/depth/segmap files
    color_0_paths = [
        os.path.join(scene_dir, f"{fnum}_colors_0.png") for fnum in range(num_images)
    ]
    color_1_paths = [
        os.path.join(scene_dir, f"{fnum}_colors_1.png") for fnum in range(num_images)
    ]
    depth_paths = [
        os.path.join(scene_dir, f"{fnum}_depth_0.png") for fnum in range(num_images)
    ]

    segmap_paths = [
        os.path.join(scene_dir, f"{fnum}_class_segmaps.png") for fnum in range(num_images)
    ]

    return color_0_paths, color_1_paths, depth_paths, segmap_paths

def create_datasets(hdf5_file, color_0_paths, depth_0_paths, segmap_paths, image_size):
    """
    Creates datasets in the HDF5 file.

    Args:
        hdf5_file (h5py._hl.files.File): Opened HDF5 file object.
        color_0_paths (List[str]): List of color_0 image paths.
        color_1_paths (List[str]): List of color_1 image paths.
        depth_0_paths (List[str]): List of depth_0 image paths.
        depth_1_paths (List[str]): List of depth_1 image paths.
        segmap_paths (List[str]): List of semantic segmentation map paths.
        image_size (tuple): Expected size of the images (height, width) in pixels.
    """
    hdf5_file.create_dataset("colors", shape=(2, len(color_0_paths)) + image_size, maxshape=(2, None) + image_size, chunks=True)
    hdf5_file.create_dataset("depth", shape=(len(depth_0_paths),)+image_size, maxshape=(None,)+image_size, chunks=True)
    hdf5_file.create_dataset("segmap", shape=(len(segmap_paths),)+image_size, maxshape=(None,)+image_size, chunks=True)


def load_and_store_data(hdf5_file, color_0_path, color_1_path, depth_0_path, segmap_path, idx, image_size):
    """
    Loads and stores data in datasets.

    Args:
        hdf5_file (h5py._hl.files.File): Opened HDF5 file object.
        color_0_path (str): Color_0 image path.
        color_1_path (str): Color_1 image path.
        depth_0_path (str): Depth_0 image path.
        depth_1_path (str): Depth_1 image path.
        segmap_path (str): Semantic segmentation map path.
        idx (int): Index of the data item.
        image_size (tuple): Expected size of the images (height, width) in pixels.
    """

    color_0 = np.array(resize_squeeze(color_0_path, image_size[1], image_size[0]))
    color_1 = np.array(resize_squeeze(color_1_path, image_size[1], image_size[0]))
    depth = np.array(resize_squeeze(depth_0_path, image_size[1], image_size[0]))
    segmap = np.array(resize_squeeze(segmap_path, image_size[1], image_size[0]))
    hdf5_file["colors"][idx][0] = color_0
    hdf5_file["colors"][idx][1] = color_1
    hdf5_file["depth"][idx] = depth
    hdf5_file["segmap"][idx] = segmap

if __name__ == "__main__":
    data_dir = "/path/to/data"
    output_dir = "/path/to/output/dir"
    image_size = (2048, 1024, 3)  # image width, image height, color channels
    convert_dir_to_hdf5(data_dir, output_dir, image_size)