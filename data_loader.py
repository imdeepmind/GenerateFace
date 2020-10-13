from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class FaceDataset(Dataset):
  def __init__(self, path, transform=None):
    self.__images = os.listdir(path)
    self.__path = path
    self.__transform = transform
  
  def __len__(self):
    return len(self.__images)
  
  def __getitem__(self, index):
    image = Image.open(os.path.join(self.__path, self.__images[index]))
    if self.__transform:
      image = self.__transform(image)

    return image