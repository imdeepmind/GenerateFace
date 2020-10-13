from torch.utils.data import Dataset
import cv2
import torch
import os

class FaceDataset(Dataset):
  def __init__(self, path, transform=None):
    self.__images = os.listdir(path)
    self.__transform = transform
  
  def __len__(self):
    return len(self.__images)
  
  def __getitem__(self, index):
    center = cv2.imread(self.__images[index], 1)

    if self.__transform:
      center = self.__transform(center)

    return center