from typing import Any, Tuple
import os

import cv2
import numpy as np
import torch
import torchvision


class TinyImageNet(torch.utils.data.Dataset):
    """
    A custom Dataset providing access to the Tiny ImageNet dataset, which is assumed to have the
    following directory structure:
    <root of dataset directory>/
        train/
            n01443537/
                n01443537_<image id>.JPEG
                ...
            <other wordnet IDs, corresponding to classes>
            ...

    There should be 200 training classes with a total of 100,000 images, and a total of 10,000 test
    images. Each directory under train/ is named for a wordnet ID that is the label of all images it
    contains.
    """

    def __init__(
        self, data_dir: str, train: bool = True, download: bool = False,
        transform: Any = None, target_transform: Any = None,
    ):
        """
        Args:
            data_dir: The path to the directory where the data is stored.
            train: Whether to use the training dataset or the test set.
            download: A dummy parameter that's required to be False.
            transform: An optional transformation to apply to the data points, i.e. the images.
            target_transform: An optional transformation to apply to the labels.
        """
        assert not download, "Downloading the dataset is not supported"

        self.transform = transform
        self.target_transform = target_transform

        if train: root = os.path.join(data_dir, "train")
        else: root = os.path.join(data_dir, "val")
        assert os.path.isdir(root), root
        self.root = root

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

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
