# data_loader.py
import os
from torchvision import datasets, transforms
import torch.utils.data as data

def get_data(train_dir, test_transforms, train_transforms, batch_size):
    # ... כאן יבוא הקוד המתוקן שכתבנו קודם ...
    train_path = os.path.join(train_dir, 'train')
    valid_path = os.path.join(train_dir, 'val')

    train_data = datasets.ImageFolder(root=train_path, transform=train_transforms)
    valid_data = datasets.ImageFolder(root=valid_path, transform=test_transforms)

    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_iterator = data.DataLoader(valid_data, batch_size=batch_size)

    return train_iterator, valid_iterator