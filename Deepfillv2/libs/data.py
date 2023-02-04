import os
import cv2
import numpy as np
import gdal

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img

def cv_loader(path):
    """ Gets as input the path of .tif files and return C x H x W shaped PIL.Image.Image in the [0, 255] range."""
    """ I am not convinced by this per-image normalization (img - img_min) / (img_max - img_min) """
    # img = cv2.imread(path, -1) # old version that uses cv2
    gdal_tile = gdal.Open(path) # new version use gdal instead of cv2
    ulx, xres, _, uly, _, yres = gdal_tile.GetGeoTransform()
    llx = ulx
    lly = uly + (gdal_tile.RasterYSize * yres)
    urx = ulx + (gdal_tile.RasterXSize * xres)
    ury = uly
    bounds = np.array([llx, lly, urx, ury]) # bounds
    img = gdal_tile.GetRasterBand(1).ReadAsArray() # values
    img_max = np.amax(img)
    img_min = np.amin(img)
    img = (img - img_min) / (img_max - img_min) # normalize all images between 0 and 1
    img = (np.stack((img,)*3, axis=-1) * 255).astype(np.uint8) # rescale from 0 to 255
    img = Image.fromarray((img))
    return img, bounds

def cv_loader_mask(path):
    mask = cv2.imread(path, -1)
    mask = np.asarray(mask, np.float32)
    mask = np.reshape(mask, (1, 256, 256))
    return mask


def is_image_file(fname):
    return fname.lower().endswith(IMG_EXTENSIONS)


class ImageDataset_segmented(Dataset):
    def __init__(self, folder_path, 
                       img_shape, 
                       random_crop=False, 
                       scan_subdirs=False, 
                       transforms=None
                       ):
        super().__init__()
        self.img_shape = img_shape
        self.random_crop = random_crop

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(
                folder_path) if is_image_file(entry.name)]

        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            print(root)
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv_loader(self.data[index])
        mask_path = self.data[index].replace('images', 'masks') # occhio che qua devo aver prima creato le training masks
        mask = cv_loader_mask(mask_path.replace('.tif', '_mask.tif'))

        if self.random_crop:
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape))(img)
            img = T.RandomCrop(self.img_shape)(img)
        else:
            img = T.Resize(self.img_shape)(img)

        img = self.transforms(img)
        img.mul_(2).sub_(1)

        return img, mask

class ImageDataset_box(Dataset):
    def __init__(self, folder_path, 
                       img_shape, 
                       random_crop=False, 
                       scan_subdirs=False, 
                       transforms=None
                       ):
        super().__init__()
        self.img_shape = img_shape # [256, 256]
        self.random_crop = random_crop

        if scan_subdirs:
            self.data = self.make_dataset_from_subdirs(folder_path)
        else:
            self.data = [entry.path for entry in os.scandir(folder_path) if is_image_file(entry.name)]

        print(f'Training images: {len(self.data)}')

        self.transforms = T.ToTensor()
        if transforms != None:
            self.transforms = T.Compose(transforms + [self.transforms])

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            print(root)
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))
        return samples  # lista dei filepaths di tutte le immagini nella cartella train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        """ Here we load the .tif images contained in the self.data list with the cv_loader method that
        returs PIL.Image.Image images. Afterwards we crop/resize and finally transform to torch tensors in [-1, 1] """

        img, bounds = cv_loader(self.data[index])  # PIL.Image.Image.

        if self.random_crop: # at the end of this if/else we will have a resized image
            w, h = img.size
            if w < self.img_shape[0] or h < self.img_shape[1]:
                img = T.Resize(max(self.img_shape))(img)
            img = T.RandomCrop(self.img_shape)(img)
        else:
            img = T.Resize(self.img_shape)(img)

        img = self.transforms(img) # yields pytorch tensor in the [0, 1] range
        img.mul_(2).sub_(1) # multiply by 2 and subtract 1 to have a [-1, 1] range

        bounds = torch.from_numpy(bounds)

        return img, bounds
