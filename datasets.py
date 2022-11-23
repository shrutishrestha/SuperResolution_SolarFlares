from os import listdir
from os.path import join
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Pad
import torchvision
import torchvision.transforms.functional as f 
import torch


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def range_normalization(image, minimum, maximum):
    return ((image - minimum) / (maximum - minimum))

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        # RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        # ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Pad(512),
        # Resize(512),
        # CenterCrop(512),
        ToTensor()
    ])


class DatasetFromFolder(Dataset):
    def __init__(self, main_folder, partition_list, crop_size, upscale_factor, train=True):
        super(DatasetFromFolder, self).__init__()

        self.partition_list = partition_list.split(',')
        self.train = train
        self.image_filenames = []
        self.main_folder = main_folder

        for partition in self.partition_list:
            self.img_folder = self.main_folder+str(partition)
            self.image_filenames += [join(self.img_folder, x) for x in listdir(self.img_folder)]
  
        self.randomcrop = RandomCrop(size= (crop_size,crop_size), fill=0)
        self.centercrop = CenterCrop(size= (crop_size,crop_size))

        self.resize = Resize(crop_size // upscale_factor, interpolation=f.InterpolationMode.BILINEAR)

        self.to_tensor = torchvision.transforms.ToTensor()
        
        self.minimum = -1500
        self.maximum = 1500
        
        
    def __getitem__(self, index):
        original = np.load(self.image_filenames[index]) #(-1500, 1500)
        original = self.to_tensor(original)
        
        if self.train:
            original_cropped = self.randomcrop(original)
            lr_cropped_resize = self.resize(original_cropped) #(-1500, 1500) #64,64
            
        else:
            original_cropped = self.centercrop(original)
            lr_cropped_resize = self.resize(original_cropped) #(-1500, 1500) #64,64  
            
        lr_norm_tensor = range_normalization(lr_cropped_resize, self.minimum, self.maximum)#(0, 1)
        hr_norm_tensor = range_normalization(original_cropped, self.minimum, self.maximum)#(0, 1)
 
        return lr_norm_tensor.float(), hr_norm_tensor.float(), original_cropped #(0, 1)  (0, 1)  

    def __len__(self):
        return len(self.image_filenames)

