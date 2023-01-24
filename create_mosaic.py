import glob
import argparse
import rioxarray
from rioxarray.merge import merge_arrays
import rasterio as rio
import xarray as xr
import geopandas as gpd

from tqdm import tqdm
from oggm import utils

from utils import contains_glacier_, rasterio_clip

"""
This program creates the whole DEM mosaic and the whole mask mosaic with all masks of single glaciers contained
in the specific RGI.
"""

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

parser = argparse.ArgumentParser(description='Create DEM mosaic from DEM tiles')
parser.add_argument("--input",  type=str, default=None, help="folder path to the DEM tiles")
parser.add_argument("--output", type=str, default="../ASTERDEM_v3_mosaics/", help="folder path for the output file")
parser.add_argument('--create_mask',  default=True, type=str2bool, const=True, nargs='?', help='Create mask mosaic')
parser.add_argument("--region",  type=str, default=None, help="RGI region in xx format")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")



def main():


    args = parser.parse_args()

    print(f'Working on tiles contained in {args.input} and creating mosaic and mask for region {args.region}')

    INDIR = args.input
    if INDIR[:-1] != '/':
            INDIR = ''.join((INDIR, '/'))

    OUTPUT = args.output

    if args.create_mask == True:
        if args.region == None:
            print("Creating mosaic mask requires RGI region: ex. --region 11")
            exit()

    # get file path of all tiles
    dem_paths = []
    for filename in glob.iglob(INDIR + '*dem.tif'):
        dem_paths.append(filename)

    src_files_to_mosaic = []
    for tile in dem_paths:
        src = rioxarray.open_rasterio(tile)
        src_files_to_mosaic.append(src)

    mosaic = merge_arrays(src_files_to_mosaic)
    mosaic.rio.to_raster(OUTPUT+f"mosaic_RGI_{args.region}.tif")
    print('Created mosaic.')

    if args.create_mask:

        utils.get_rgi_dir(version=args.version)
        eu = utils.get_rgi_region_file(args.region, version=args.version)
        gdf = gpd.read_file(eu)

        print(f"Creating mosaic mask ... should be able to burn in {len(gdf)} glaciers")

        """ 
        OLD METHOD 
        old version which first selects for every tile those glaciers contained, then produces all invidual mask tiles 
        and then merges them into a mosaic mask. Unless you want to have individual mask tiles produces, you can just
        create an empty mosaic and all glaciers contained inside will be burned in. 
         
        for tile in tqdm(dem_paths, leave=True):
            glacier_frame = contains_glacier_(tile, gdf, .5)
            glaciers_alps = sum(glacier_frame['RGIId'].tolist(), [])
            boolean_series = gdf['RGIId'].isin(glaciers_alps)
            filtered_gdf = gdf[boolean_series]
            filtered_gdf = filtered_gdf.reset_index()
            # NB we can avoid the loop and simply call rasterio_clip(tile, gdf, args.epsg)
            _ = rasterio_clip(tile, filtered_gdf, args.epsg)

        src_files_to_mask = []
        for tile in dem_paths:
            src = rioxarray.open_rasterio(tile.replace('.tif', '_mask.tif'))
            src_files_to_mask.append(src)

        # here it appears necessary to pass bounds=mosaic.rio.bounds() to avoid one extra column and row to be added
        mask = merge_arrays(src_files_to_mask, bounds=mosaic.rio.bounds())
        mask.rio.to_raster(OUTPUT.replace('.tif', '_mask.tif'))"""

        """ METHOD 1: MASK SINGLE TILES AND MERGE EVERYTHING TO CREATE MOSAIC MASK"""
        use_method1 = True
        if use_method1 is True:
            src_files_to_mask = []
            for tile in tqdm(dem_paths, leave=True):
                masked_tile = rioxarray.open_rasterio(tile)
                masked_tile = xr.zeros_like(masked_tile)
                masked_tile.rio.write_nodata(1., inplace=True)
                masked_tile = masked_tile.rio.clip(gdf['geometry'].to_list(), args.epsg, drop=False, invert=True, all_touched=False)
                #num_px_burned = float(masked_tile.sum()) # if tile is useless num_px_burned=0
                #if (num_px_burned==0): print(tile)
                masked_tile.rio.write_nodata(0., inplace=True) # necessary to fill in missing tiles with 0 when merging
                src_files_to_mask.append(masked_tile)
                # masked_tile.rio.to_raster(dem_path[:-4] + "_mask.tif") # if you want to save the masked_tile
            mask1 = merge_arrays(src_files_to_mask, bounds=mosaic.rio.bounds())
            mask1.rio.to_raster(OUTPUT + f"mosaic_RGI_{args.region}.tif".replace('.tif', '_mask.tif'))


        """ METHOD 2: FAST way is burn in all glaciers in one go. Memory very expensive """
        use_method2 = False
        if use_method2 is True:
            mask2 = xr.zeros_like(mosaic)
            mask2.rio.write_nodata(1., inplace=True)
            mask2 = mask2.rio.clip(gdf['geometry'].to_list(), args.epsg, drop=False, invert=True, all_touched=False)
            mask2.rio.to_raster(OUTPUT+f"mosaic_RGI_{args.region}.tif".replace('.tif', '_mask.tif'))

        if (use_method1 is True and use_method2 is True):
            print(mask1.equals(mask2)) # check if the two methods agree

        print("Finished !")

if __name__ == '__main__':
    main()