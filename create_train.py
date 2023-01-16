import os
import argparse
import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt

from set_generation import flow_train_dataset




parser = argparse.ArgumentParser(description='Create training images')
parser.add_argument("--input",  type=str, default=None, help="path to the DEM tile folder")
parser.add_argument("--outdir", type=str, default="dataset/", help="path for the output file")

# currently not used
parser.add_argument("--region",  type=int, default=11, help="RGI region")
parser.add_argument("--shape",  type=int, default=256, help="Size of test patches")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")

# adjust naming and sampling behavior 
parser.add_argument("--max_height",  type=int, default=4800, help="Max desired height of training samples")
parser.add_argument("--threshold",  type=int, default=2000, help="Threshold value to sample high elevation regions")
parser.add_argument("--mode",  type=str, default='average', help="Threshold mode: average or max")
parser.add_argument("--samples",  type=int, default=25000, help="Number of samples to attempt to create")
parser.add_argument("--postfix",  type=str, default='a', help="postfix added behind samples")


def main():

    args = parser.parse_args()

    # TODO: modificare questi path in modo da tenere traccia degli iperparametri
    # 'train/' diventa 'train/RGI_{args.region}_size_{args.shape}_maxH_{args.max_height}_eccetera'
    train_path = args.outdir + 'train/'
    val_path = args.outdir + 'val/'

    if os.path.isdir(train_path):
        if os.path.isdir(train_path + 'images/'):
            None
        else:
            os.mkdir(train_path + 'images/')
    else:
        os.mkdir(train_path)
        os.mkdir(train_path + 'images/')

    if os.path.isdir(val_path):
        if os.path.isdir(val_path + 'images/'):
            None
        else:
            os.mkdir(val_path + 'images/')
    else:
        os.mkdir(val_path)
        os.mkdir(val_path + 'images/')


    # load DEM and glacier mask
    dem = rioxarray.open_rasterio(args.input).squeeze()
    empty = rioxarray.open_rasterio(args.input.replace('.tif', '_mask.tif')).squeeze()

    print("Attempting to create {} samples".format(args.samples))

    _ = flow_train_dataset(dem, empty, (args.shape, args.shape),
                           train_path+'images/', val_path+'images/',
                           args.max_height, args.threshold, args.mode, args.samples, args.postfix)




if __name__ == '__main__':
    main()