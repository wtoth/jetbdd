import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image

class BSDD_Dataset(Dataset):
    def __init__(self, dataset, spatial_transforms=None, color_transforms=None):
        self.image_directory = pd.read_csv(dataset)
        self.spatial_transforms = spatial_transforms
        self.color_transforms = color_transforms

    def __len__(self) -> int:
        return len(self.image_directory)
    
    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("images")]
        mask_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("labels")]
        image = Image.open(image_path)
        mask = Image.open(mask_path)


        if self.spatial_transforms:
            # spatial transforms applied to both
            image, mask = self.spatial_transforms(image, mask)

        if self.color_transforms:
            # color transforms applied only to image
            image = self.color_transforms(image)

        # normalize images
        image = image.float() / 255.0
        mask = mask.float() / 255.0 

        weight_map = self.compute_weight_map(mask)

        return image, mask, weight_map