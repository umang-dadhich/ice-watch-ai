import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_aug(img_size=512):
    return A.Compose([
        A.RandomCrop(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
        A.GaussNoise(p=0.15),
        ToTensorV2(transpose_mask=True),
    ])


def build_val_aug(img_size=512):
    return A.Compose([
        A.CenterCrop(img_size, img_size, always_apply=True),
        ToTensorV2(transpose_mask=True),
    ])
