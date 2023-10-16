import os
import sys
import argparse
import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
from set_generation import flow_train_dataset


parser = argparse.ArgumentParser(description='Create training images')
parser.add_argument("--input",  type=str, default=None, help="input DEM mosaic file")
parser.add_argument("--outdir", type=str, default="dataset/", help="path for the output files")
parser.add_argument("--region",  type=str, default=None, help="RGI region")
parser.add_argument("--shape",  type=int, default=384, help="size of train patches") # 256
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")

# adjust naming and sampling behavior 
parser.add_argument("--max_height",  type=int, default=9999, help="Max desired height of training samples")
parser.add_argument("--threshold",  type=int, default=2500, help="Threshold value to sample high elevation regions")
parser.add_argument("--mode",  type=str, default='average', help="Threshold mode: average or max")
parser.add_argument("--samples",  type=int, default=40, help="Number of samples to attempt to create") # 40000
parser.add_argument("--postfix",  type=str, default='', help="postfix added behind samples")


def main():

    args = parser.parse_args()

    if args.region == None:
        print("Creating mosaic mask requires RGI region: ex. --region 11")
        exit()

    train_path = args.outdir + 'train/' + f'RGI_{args.region}_{args.mode}_{args.threshold}_size_{args.shape}/'
    val_path = args.outdir + 'val/' + f'RGI_{args.region}_{args.mode}_{args.threshold}_size_{args.shape}/'
    print(f'Output train images: {train_path}')
    print(f'Output val images: {val_path}')

    if os.path.isdir(train_path):
        if os.path.isdir(train_path + 'images/'):
            None
        else:
            os.makedirs(train_path + 'images/')
    else:
        os.makedirs(train_path)
        os.makedirs(train_path + 'images/')

    if os.path.isdir(val_path):
        if os.path.isdir(val_path + 'images/'):
            None
        else:
            os.makedirs(val_path + 'images/')
    else:
        os.makedirs(val_path)
        os.makedirs(val_path + 'images/')

    # load DEM and glacier mask
    dem = rioxarray.open_rasterio(args.input).squeeze()
    if (args.region == '13' or args.region == '14' or args.region == '15'):
        print('merging the three mosaic masks 13-14-15...')
        m13 = rioxarray.open_rasterio('/media/nico/samsung_nvme/ASTERDEM_v3_mosaics/mosaic_RGI_13_mask.tif')
        m14 = rioxarray.open_rasterio('/media/nico/samsung_nvme/ASTERDEM_v3_mosaics/mosaic_RGI_14_mask.tif')
        m15 = rioxarray.open_rasterio('/media/nico/samsung_nvme/ASTERDEM_v3_mosaics/mosaic_RGI_15_mask.tif')
        empty = (m13 + m14 + m15).squeeze()
    else:
        empty = rioxarray.open_rasterio(args.input.replace('.tif', '_mask.tif')).squeeze()

    print("Attempting to create {} samples".format(args.samples))

    _ = flow_train_dataset(dem, empty, args.region, (args.shape, args.shape),
                           train_path+'images/', val_path+'images/',
                           args.max_height, args.threshold, args.mode, args.samples, args.postfix)




if __name__ == '__main__':
    main()