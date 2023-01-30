import os
import glob
import cv2
import gc
import argparse

import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
import rioxarray
import xarray as xr

import matplotlib.pyplot as plt

from oggm import utils
from tqdm import tqdm

from utils import coords_to_xy, contains_glacier_


parser = argparse.ArgumentParser(description='Create DEM mosaic from DEM tiles')
parser.add_argument("--input",  type=str, default='../ASTERDEM_v3_mosaics/', help="path for input mosaic file")
parser.add_argument("--outdir", type=str, default="dataset/test/", help="path for the output test files")
parser.add_argument("--region",  type=str, default=None, help="RGI region in xx format")
parser.add_argument("--shape",  type=int, default=256, help="Size of test patches")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")




def main():

    args = parser.parse_args()

    args.outdir = args.outdir+f'RGI_{args.region}_size_{args.shape}/'

    if os.path.isdir(args.outdir):
        if os.path.isdir(args.outdir + 'images/'):
            None
        else:
            os.mkdir(args.outdir + 'images/')
        if os.path.isdir(args.outdir + 'masks/'):
            None
        else:
            os.mkdir(args.outdir + 'masks/')
        if os.path.isdir(args.outdir + 'masks_full/'):
            None
        else:
            os.mkdir(args.outdir + 'masks_full/')
    else:
        os.mkdir(args.outdir)
        os.mkdir(args.outdir + 'images/')
        os.mkdir(args.outdir + 'masks/')
        os.mkdir(args.outdir + 'masks_full/')

    # NM: useful links.
    # https://gis.stackexchange.com/questions/353698/how-to-clip-an-xarray-to-a-smaller-extent-given-the-lat-lon-coordinates
    # https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.rioxarray.XRasterBase.isel_window
    # https://gis.stackexchange.com/questions/299787/finding-pixel-location-in-raster-using-coordinates

    # fetch RGI
    utils.get_rgi_dir(version=args.version)
    eu = utils.get_rgi_region_file(args.region, version=args.version)
    gdf = gpd.read_file(eu)

    # load DEM
    args.input = args.input+f'mosaic_RGI_{args.region}.tif'
    dem = rioxarray.open_rasterio(args.input)
    if (args.region =='13' or args.region =='14' or args.region =='15'):
        print('merging the three mosaic masks 13-14-15...')
        m13 = rioxarray.open_rasterio('/home/nico/PycharmProjects/skynet/ASTERDEM_v3_mosaics/mosaic_RGI_13_mask.tif')
        m14 = rioxarray.open_rasterio('/home/nico/PycharmProjects/skynet/ASTERDEM_v3_mosaics/mosaic_RGI_14_mask.tif')
        m15 = rioxarray.open_rasterio('/home/nico/PycharmProjects/skynet/ASTERDEM_v3_mosaics/mosaic_RGI_15_mask.tif')
        empty = m13 + m14 + m15
    else:
        empty = rioxarray.open_rasterio(args.input.replace('.tif', '_mask.tif'))

    # sort glaciers
    # da queste righe estraggo filtered_gdf, la lista di ghiacciai contenuti nel dem completo
    glacier_frame = contains_glacier_(args.input, gdf, 0.)
    glaciers_alps = sum(glacier_frame['RGIId'].tolist(), [])
    boolean_series = gdf['RGIId'].isin(glaciers_alps)
    filtered_gdf = gdf[boolean_series]
    filtered_gdf = filtered_gdf.reset_index()

    # da queste righe ricalcolo il centro dei ghiacciai e li sovrascrivo in filtered_gdf
    print("Creating new glacier center points:")
    for i in tqdm(range(len(filtered_gdf))):
        geometry = filtered_gdf['geometry'][i]
        lon_min, lat_min, lon_max, lat_max = geometry.bounds
        longitude, latitude = np.mean([lon_min,lon_max]), np.mean([lat_min,lat_max])
        filtered_gdf.loc[i, 'CenLon'] = longitude # note that i think this replaces the default values
        filtered_gdf.loc[i, 'CenLat'] = latitude # note that i think this replaces the default values

    # convert lat/lon to x/y for images
    print('Translating lat/lon to x/y ...')
    coords, RGI = coords_to_xy(args.input, filtered_gdf)
    coords_frame = pd.DataFrame({'RGIId': RGI, 'rows': coords[:,0], 'cols': coords[:,1]})
    rows = np.array(coords_frame['rows'].tolist())
    cols = np.array(coords_frame['cols'].tolist())
    RGIId = coords_frame['RGIId'].tolist()
    print('Done.')


    mask = rioxarray.open_rasterio(args.input.replace('.tif', '_mask.tif'))
    mask = xr.zeros_like(mask)
    mask_copy = mask

    resolution = float(mask.coords['x'][1] - mask.coords['x'][0])  # calculated on x-axis.
    print(f"Raster resolution: {resolution}")


    with tqdm(total=len(coords_frame), leave=True) as progress:
        for glacier in range(len(coords_frame)):
            progress.set_postfix_str(RGIId[glacier])

            geom = filtered_gdf['geometry'][glacier]
            lon_min, lat_min, lon_max, lat_max = geom.bounds
            dx, dy = (lat_max - lat_min) / resolution, (lon_max - lon_min) / resolution

            mask.rio.write_nodata(1, inplace=True)

            # glacier too small
            if (dx==0 or dy==0):
                tqdm.write(f"{RGIId[glacier]} with bounding box ({int(dx)},{int(dy)}) has been excluded.")

            # glacier too big. Note this is something that should be addressed in the future.
            elif (dx > args.shape or dy > args.shape):
                tqdm.write(f"{RGIId[glacier]} with bounding box ({int(dx)},{int(dy)}) has been excluded.")

            # glacier OK
            else:
                r = rows[glacier] - int(args.shape/2) if rows[glacier] >= int(args.shape/2) else rows[glacier]
                c = cols[glacier] - int(args.shape/2) if cols[glacier] >= int(args.shape/2) else cols[glacier]

                # note that mask_patch, image_patch and full_mask are xarray.core.dataarray.DataArray
                ########### extract mask patch #############
                # First reduce mask then clip the geometry
                mask_patch = mask[0, r:r + args.shape, c:c + args.shape]
                mask_patch = mask_patch.rio.clip([geom], "EPSG:4326", drop=False, invert=True, all_touched=False)
                ########### extract image patch #############
                image_patch = dem[0, r:r + args.shape, c:c + args.shape]
                ########### extract full mask #############
                full_mask = empty[0, r:r + args.shape, c:c + args.shape]



                show_some = False
                if (show_some and glacier % 100 == 0):
                    fig, (ax0, ax1, ax2) = plt.subplots(3,1, figsize=(4.5,10))
                    #im0 = ax0.imshow(image_patch, cmap='terrain')
                    image_patch.plot.imshow(ax=ax0, cmap='terrain', cbar_kwargs={'label': 'Height (m)'})
                    ax0.plot(*geom.exterior.xy, c='red')
                    ax0.set_title('image')
                    ax0.set_xlabel('')
                    ax0.set_ylabel('')
                    mask_patch.plot.imshow(ax=ax1, cmap='Blues', cbar_kwargs={'label': ''})
                    ax1.set_title('mask')
                    ax1.set_xlabel('')
                    ax1.set_ylabel('')
                    full_mask.plot.imshow(ax=ax2, cmap='Blues', cbar_kwargs={'label': ''})
                    ax2.set_title('masks_full')
                    ax2.set_ylabel('')
                    ax2.set_xlabel(RGIId[glacier], fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    plt.show()

                    fig = plt.figure(figsize=(6, 5))
                    ax0 = fig.add_subplot(111, projection='3d')
                    image_patch.plot.surface(ax=ax0, cmap='terrain', add_colorbar=False)
                    ax0.plot(*geom.exterior.xy, c='red')
                    ax0.set_title(RGIId[glacier], fontsize=13, fontweight='bold')
                    ax0.set_xlabel('')
                    ax0.set_ylabel('')
                    ax0.set_zlabel('Height (m)')
                    plt.tight_layout()
                    plt.show()

                ## save ndarray as tif files (not georeferenced)
                ## cv2.imwrite(args.outdir + 'masks/' + RGIId[glacier] + '_mask.tif', mask_patch.to_numpy().astype(np.float32))
                ## cv2.imwrite(args.outdir + 'images/' + RGIId[glacier] + '.tif', image_patch.to_numpy().astype(np.uint16))
                ## cv2.imwrite(args.outdir + 'masks_full/' + RGIId[glacier] +'_mask.tif', full_mask.to_numpy().astype(np.float32))

                # save xarray.DataArray as tif files
                mask_patch.rio.to_raster(args.outdir + 'masks/' + RGIId[glacier] + '_mask.tif', dtype=np.float32)
                image_patch.rio.to_raster(args.outdir + 'images/' + RGIId[glacier] + '.tif', dtype=np.uint16)
                full_mask.rio.to_raster(args.outdir + 'masks_full/' + RGIId[glacier] + '_mask_full.tif', dtype=np.float32)

            # remove mask
            mask.rio.write_nodata(0, inplace=True)
            mask = mask_copy
            progress.update()
            gc.collect()

if __name__ == '__main__':
    main()