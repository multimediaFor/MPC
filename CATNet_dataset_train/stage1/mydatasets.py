import warnings
import cv2
import random
import numpy as np
import albumentations as A
import torch
from torchvision import transforms
from torch.utils.data import Dataset


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


class MyDataset(Dataset):
    def __init__(self, num=0, file='', choice='train', input_size=512, gt_ratio=4):
        self.num = num
        self.choice = choice
        self.filelist = file
        self.gt_ratio = gt_ratio

        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
        ])
        self.size = input_size
        self.albu = A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.0), p=0.75),
            A.PadIfNeeded(min_height=self.size, min_width=self.size, p=1.0),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
            ], p=0.75),
            A.ImageCompression(quality_lower=50, quality_upper=95, p=0.75),
            A.OneOf([
                A.OneOf([
                    A.Blur(p=1),
                    A.GaussianBlur(p=1),
                    A.MedianBlur(p=1),
                    A.MotionBlur(p=1),
                ], p=1),
                A.OneOf([
                    A.Downscale(p=1),
                    A.GaussNoise(p=1),
                    A.ISONoise(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.RandomToneCurve(p=1),
                    A.Sharpen(p=1),
                ], p=1),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1),
                    A.GridDistortion(p=1),
                ], p=1),
            ], p=0.25),
        ], is_check_shapes=False)

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.num

    def load_item(self, idx):
        fname1, fname2 = self.filelist[idx]

        img = cv2.imread(fname1)[..., ::-1]
        H, W, _ = img.shape
        mask = cv2.imread(fname2) if fname2 != '' else np.zeros([H, W, 3])
        mask = thresholding(mask)

        if self.choice == 'train' and random.random() < 0.75:
            aug = self.albu(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']

        img = cv2.resize(img, (self.size, self.size))
        img = img.astype('float') / 255.

        if self.choice == 'train':
            mask = cv2.resize(mask, (self.size // self.gt_ratio, self.size // self.gt_ratio))
            mask = thresholding(mask)

        mask = mask.astype('float') / 255.
        mask = self.tensor(mask[:, :, :1])
        return self.transform(img), mask, H, W, fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)
