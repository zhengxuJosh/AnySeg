import os
from turtle import width
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
from torch.utils.data import DataLoader
# from .augmentations_mm import get_train_augmentation
import cv2
from PIL import Image
import torch.nn.functional as F

class MUSES(Dataset):
    """
    num_classes: 19
    """
    CLASSES = [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
        'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
    ]

    PALETTE = torch.tensor([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
        [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
    ])
    
    def __init__(self, root: str = '/mnt/dev-ssd-8T/zheng/data/muses/multimodal', split: str = 'train', transform=None, modals=['frame_camera'], weather_condition=None, time_of_day=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.files = sorted(glob.glob(os.path.join(root,'frame_camera', split, '*', '*', '*.png')))
        
        if weather_condition is not None:
            if weather_condition in ['clear', 'fog', 'rain', 'snow']:
                _temp_files = [f for f in self.files if weather_condition in f]
                self.files = _temp_files
            else:
                raise Exception("Weather condition not available.")
        else:
            # Get all weather conditions if none provided
            self.files = [f for f in self.files if any(cond in f for cond in ['clear', 'fog', 'rain', 'snow'])]
        if time_of_day is not None:
            if time_of_day in ['day', 'night']:
                _temp_files = [f for f in self.files if time_of_day in f]
                self.files = _temp_files
            else:
                raise Exception("Time of day not available.")
        else:
            # Get all times of day if none provided
            self.files = [f for f in self.files if any(tod in f for tod in ['day', 'night'])]
        if not self.files:
            raise Exception(f"No images found in {root}")
        print(f"Found {len(self.files)} {split} images for weather condition '{weather_condition}' and time of day '{time_of_day}'.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        base_path = str(self.files[index])
        sample = {}
        H, W = None, None
        
        # Load each modality specified
        for modal in self.modals:
            img_path = base_path.replace('frame_camera', f'{modal}')

            # Use PIL to open the image
            img = Image.open(img_path)
            
            # Convert the image to RGB if it isn't already
            img = img.convert('RGB')
            
            # Resize the image (replace width and height with your desired dimensions)
            new_size = (1024, 1024)  # Specify the new size as a tuple
            img = img.resize(new_size)
            
            # Convert the image to a numpy array
            img = np.array(img)
            
            # If the image is in uint16 format, scale it down to uint8
            if img.dtype == 'uint16':
                img = (img / 256).astype('uint8')
            
            # Convert the image to a tensor and permute the dimensions
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            
            # Assign the tensor to the sample dictionary
            sample[modal] = img_tensor
            H, W = sample[modal].shape[1:]
        
        # Load ground truth label
        lbl_path = base_path.replace('/frame_camera', '/gt_semantic').replace('frame_camera', 'gt_labelTrainIds')
        
        # image = Image.open(lbl_path).convert('L') 
        # image = image.resize(new_size)

        # image_array = np.array(image)
        # label = torch.from_numpy(image_array).long().unsqueeze(0) 

        label = io.read_image(lbl_path)[0, ...].unsqueeze(0)
        label = F.interpolate(label.unsqueeze(0), size=new_size, mode='nearest').squeeze(1).long()
        
        # print(label.size())
        # print(torch.unique(label))

        label[label == 255] = 0
        sample['mask'] = label
        sample['img'] = sample['frame_camera']
        # Apply transformations if any
        # if self.transform:
        #     sample = self.transform(sample)
        label = sample['mask']
        sample['frame_camera'] = sample['img']
        del sample['mask']
        del sample['img']
        # label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        return sample, label.squeeze(0)


if __name__ == '__main__':
    cases = ['clear', 'fog', 'rain', 'snow']
    modals = ['frame_camera', 'event_camera', 'lidar', 'radar']
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    for case in cases:
        trainset = MUSES(transform=traintransform, modals=modals, split='val', weather_condition=case, time_of_day='day')
        trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

        for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))