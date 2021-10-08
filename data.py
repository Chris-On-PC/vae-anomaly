import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import numpy as np
import cv2

class XrayImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform: Optional[Callable] = None, patch_size: int = 128, overlap: float = 0.25):
        #super(XrayImageDataset, self).__init__(img_dir, transform=transform, patch_size = patch_size, overlap = overlap)

        self.img_dir = img_dir
        self.transform = transform
        self.overlap = overlap
        self.patch_size = patch_size
        self.filenames = os.listdir(self.img_dir)


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.filenames[idx]))
        patches = self.split_image(image)
        if self.transform:
            patches = self.transform(patches)

        return patches


    def split_image(self, image):
        img_h, img_w, img_d = image.shape

        X_points = self.start_points(img_w, self.patch_size, self.overlap)
        Y_points = self.start_points(img_h, self.patch_size, self.overlap)
        count = 0
        frmt = "png"
        patches = np.empty((len(Y_points)*len(X_points), self.patch_size, self.patch_size, img_d))
        for i in Y_points:
            for j in X_points:
                split = image[i:i+self.patch_size, j:j+self.patch_size]

                patches[count, :, :, :] = split
                cv2.imwrite('patch_{}.{}'.format( count, frmt), split)
                count += 1
        return patches

    
    def start_points(self, size, patch_size, overlap=0):
        points = [0]
        stride = int(patch_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + patch_size >= size:
                points.append(size - patch_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points