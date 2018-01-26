import numpy as np
import random
from src.dataset import LabeledImageDataset, LabeledImageDatasetBuilder


class SortedImageDataSet(LabeledImageDataset):
    def __init__(self, labeled_images):
        super().__init__(labeled_images)
        self.sorted_images = self.sort_images(labeled_images)

    def get_triplet(self):
        labels = list(self.sorted_images.keys())
        anc_label = np.random.choice(labels)
        anc_label_len = len(self.sorted_images[anc_label])
        anc = self.sorted_images[anc_label][np.random.randint(anc_label_len)]
        pos = self.sorted_images[anc_label][np.random.randint(anc_label_len)]
        while(anc == pos):
            pos = self.sorted_images[anc_label][np.random.randint(anc_label_len)]

        labels.remove(anc_label)
        neg_label = np.random.choice(labels)
        neg_label_len = len(self.sorted_images[neg_label])
        neg = self.sorted_images[neg_label][np.random.randint(neg_label_len)]
        return ([anc()], [pos()], [neg()])

    def sort_images(self, labeled_images):
        label_mapping = {}
        for lbd_img, label in labeled_images:
            label_mapping[label] = label_mapping.get(label, []) + [lbd_img]
        return label_mapping


class SortedImageDatasetBuilder(LabeledImageDatasetBuilder):
    def get_sorted_image_dataset(self):
        return SortedImageDataSet(self.images)

    def get_sorted_image_dataset_split(self, splitsize):
        splitnumber = int(round(len(self.images) * splitsize))
        dataset1 = SortedImageDataSet(self.images[:splitnumber])
        dataset2 = SortedImageDataSet(self.images[splitnumber:])
        return dataset1, dataset2
