from typing import Any, Tuple
import os

import cv2
import numpy as np
import torch
import torchvision


class ImageNet(torch.utils.data.Dataset):
    """
    A custom Dataset providing access to the ImageNet dataset or any of its subsets, such as Tiny
    ImageNet (a sample of 200 classes) or ImageNet-1K (a sample of 1,000 classes). The dataset
    is assumed to have the following directory structure:
        <root of dataset directory>/
            train/
                n01443537/
                    n01443537_<image id>.JPEG
                    ...
                <other wordnet IDs, corresponding to classes>
                ...
            val/
                <same structure as train/>

    Each directory under train/ is named for a wordnet ID that is the label of all images it
    contains.
    """

    def __init__(
        self, data_dir: str, train: bool = True, download: bool = False,
        transform: Any = None, target_transform: Any = None, use_file_list: bool = True,
    ):
        """
        Args:
            data_dir: The path to the directory where the data is stored.
            train: Whether to use the training dataset or the test set.
            download: A dummy parameter that's required to be False.
            transform: An optional transformation to apply to the data points, i.e. the images.
            target_transform: An optional transformation to apply to the labels.
            use_file_list: If true, looks for a file containing on each line a path (relative to
                the data directory) to each image file. This file is assumed to be named "train.txt"
                or "val.txt", depending on <train>, in the data directory. If provided, this can
                dramatically speed up initialization of this dataset.
        """
        assert not download, "Downloading the dataset is not supported"

        self.transform = transform
        self.target_transform = target_transform

        if train: root = os.path.join(data_dir, "train")
        else: root = os.path.join(data_dir, "val")
        assert os.path.isdir(root), root
        self.root = root

        if use_file_list:
            if train: file_list = "train.txt"
            else: file_list = "val.txt"

            with open(os.path.join(data_dir, file_list), "r") as f:
                lines = f.read().splitlines()

            wnid_dirs, self.images = zip(*[image_path.split("/") for image_path in lines])
            self.wordnet_ids = list(set(wnid_dirs))

            wnid_to_label = { wnid: label for label, wnid in enumerate(self.wordnet_ids) }
            self.targets = [wnid_to_label[wnid] for wnid in wnid_dirs]
        else:
            # We use a wordnet id's index in this list as the label.
            self.wordnet_ids = os.listdir(root)

            self.images, self.targets = zip(*[
                (image_file, label)
                for label, wordnet_id in enumerate(self.wordnet_ids)
                for image_file in os.listdir(os.path.join(root, wordnet_id))
            ])

    # Override.
    def __len__(self) -> int:
        """
        Returns:
            The total number of (image, label) pairs in this dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Given an index, retrieves the corresponding image (as a numpy tensor) and label. The label
        is not the wordnet ID but an assigned integral index.

        Args:
            index: The index.

        Returns:
            The corresponding (image, label) pair.
        """
        image_file, label = self.images[index], self.targets[index]
        image = cv2.imread(os.path.join(self.root, self.wordnet_ids[label], image_file))
        assert image is not None, image_file

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
