import sys
import os
sys.path.insert(0, os.getcwd())

from typing import Dict, Tuple

import PIL.Image
import numpy as np
import torch
import remote_images
import torchvision.transforms as transforms


def _prepare_image(image: PIL.Image.Image) -> torch.Tensor:
    img_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    img = img_transforms(image)
    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img = img.float()
        input = img.unsqueeze(0)\
            .sub_(mean)\
            .div_(std)
    return input.squeeze()


def _mask_to_class_indexes(mask: np.ndarray):
    return np.nonzero(mask)[0]


class TaggingDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data: Dict[str, np.ndarray],
                 split: str = 'train'):
        self.data = data
        self.split = split

    def __iter__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for image_id, mask in self.data.items():
            image = remote_images.load_image(image_id, self.split)
            # yield _prepare_image(image), torch.from_numpy(_mask_to_class_indexes(mask))
            yield _prepare_image(image), torch.from_numpy(mask).float()
