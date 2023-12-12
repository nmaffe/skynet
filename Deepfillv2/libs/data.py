import os
import cv2
import numpy as np
import gdal
# from PIL import Image
from utils import haversine
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

# useless
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        return img

# useless
def cv_loader(path):
    """ Gets as input the path of .tif files and return C x H x W shaped PIL.Image.Image in the [0, 255] range."""
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

def img_loader(path):
    gdal_tile = gdal.Open(path)  # new version use gdal instead of cv2
    ulx, xres, _, uly, _, yres = gdal_tile.GetGeoTransform()
    llx = ulx
    lly = uly + (gdal_tile.RasterYSize * yres)
    urx = ulx + (gdal_tile.RasterXSize * xres)
    ury = uly
    bounds = np.array([llx, lly, urx, ury])  # bounds
    img = gdal_tile.GetRasterBand(1).ReadAsArray()  # values
    return img, bounds, xres

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

# useless
class ImageDataset_box_old(Dataset):
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

def get_transforms(config, data='train'):

    # extract some useful stuff from train.yaml
    H, W = config.img_shapes[:2]
    p_random_crop = config.random_crop
    p_horflip = config.random_horizontal_flip
    p_verflip = config.random_vertical_flip
    h_min_crop = config.random_crop_hmin
    h_max_crop = config.random_crop_hmax

    if data == 'train':
        return A.Compose([
            A.HorizontalFlip(p=p_horflip),
            A.VerticalFlip(p=p_verflip),
            A.SmallestMaxSize(max(H, W), p=0.0),
            # A.ShiftScaleRotate an option ?
            A.RandomSizedCrop(min_max_height=[h_min_crop, h_max_crop], height=H, width=W,
                              w2h_ratio=1.0, p=p_random_crop),
            #A.RandomCrop(H, W, p=p_random_crop),
            #A.Resize(H, W),
            ToTensorV2()
        ])

    elif data == 'val':
        return A.Compose([
            A.Resize(H, W, p=0.0),
            ToTensorV2()
        ])

class ImageDataset_box(Dataset):
    def __init__(self, folder_path,
                 img_shape,
                 scan_subdirs=False,
                 transforms=None
                 ):
        super().__init__()
        self.img_shape = img_shape  # [256, 256]

        self.data_all_folders = []

        if scan_subdirs:
            for ifolder in folder_path:
                self.data = self.make_dataset_from_subdirs(ifolder) #folder_path
                self.data_all_folders.extend(self.data)

        else:
            for ifolder in folder_path:
            #self.data = [entry.path for entry in os.scandir(folder_path) if is_image_file(entry.name)]
                self.data = [entry.path for entry in os.scandir(ifolder) if is_image_file(entry.name)]
                print(f'Folder: {ifolder}, No. images: {len(self.data)}')
                self.data_all_folders.extend(self.data)

        print(f'Total number of images: {len(self.data_all_folders)}')

        self.transforms = transforms

    def make_dataset_from_subdirs(self, folder_path):
        samples = []
        for root, _, fnames in os.walk(folder_path, followlinks=True):
            for fname in fnames:
                if is_image_file(fname):
                    samples.append(os.path.join(root, fname))
            if len(samples)>0:
                print(f'Folder: {root}, No. images: {len(samples)}')
            else:
                print(f'Folder: {root}')
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img, bounds, ris_ang = img_loader(self.data[index])  # img ndarray uint16 (256, 256)
        img = img.astype(np.float32)  # convert to float32

        lon_c = 0.5 * (bounds[0] + bounds[2])
        lat_c = 0.5 * (bounds[1] + bounds[3])
        ris_metre_lon = haversine(lon_c, lat_c, lon_c + ris_ang, lat_c) * 1000  # m
        ris_metre_lat = haversine(lon_c, lat_c, lon_c, lat_c + ris_ang) * 1000  # m

        # Note: since the transformations include cropping, the original bounds do not correspond to the transformed image.
        img = self.transforms(image=img)['image'] # (1, 256, 256)
        # Calculate slope
        slope_lat, slope_lon = torch.gradient(img, spacing=[ris_metre_lat, ris_metre_lon], dim=(1, 2))  # (1, 256, 256)

        # Normalize dem
        # normalize to [0, 1] and scale to [-1, 1]
        img_max = torch.max(img)  #9000.0
        img_min = 0.0  # torch.min(img)
        img = (img - img_min) / (img_max - img_min) # note that this can cause nans
        img.mul_(2).sub_(1) # (1, 256, 256)

        # Normalize slopes
        # Normalize to [0, 1] and scale to [-1, 1]
        # maxs_lat = torch.amax(slope_lat, dim=(1, 2)).unsqueeze(1).unsqueeze(2)  # (1,1,1)
        # mins_lat = torch.amin(slope_lat, dim=(1, 2)).unsqueeze(1).unsqueeze(2)  # (1,1,1)
        # maxs_lon = torch.amax(slope_lon, dim=(1, 2)).unsqueeze(1).unsqueeze(2)  # (1,1,1)
        # mins_lon = torch.amin(slope_lon, dim=(1, 2)).unsqueeze(1).unsqueeze(2)  # (1,1,1)
        # print(mins_lat, maxs_lat, mins_lon, maxs_lon)
        mins_lat, maxs_lat = -10., 10.
        mins_lon, maxs_lon = -10., 10.

        # normalization slopes
        slope_lat = torch.clip(slope_lat, min=-10., max=10.)
        slope_lon = torch.clip(slope_lon, min=-10., max=10.)
        slope_lat = (slope_lat - mins_lat) / (maxs_lat - mins_lat)  # (1, 256, 256)
        slope_lon = (slope_lon - mins_lon) / (maxs_lon - mins_lon)  # (1, 256, 256)
        slope_lat.mul_(2).sub_(1)
        slope_lon.mul_(2).sub_(1)

        #img = img.repeat(3, 1, 1)  # convert to (3, 256, 256) old 3-ch

        # To avoid singularities in the normalizations we replace nans with 0s
        img = torch.nan_to_num(img)             # (3, 256, 256)
        slope_lat = torch.nan_to_num(slope_lat) # (1, 256, 256)
        slope_lon = torch.nan_to_num(slope_lon) # (1, 256, 256)

        return img, slope_lat, slope_lon, img_min, img_max, ris_metre_lon, ris_metre_lat, torch.from_numpy(bounds)