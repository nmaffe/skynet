import gc
import sys
from tqdm import tqdm
import os, glob
import argparse
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rioxarray.merge import merge_arrays
import matplotlib
import matplotlib.pyplot as plt

def fetch_dem(folder_mosaic=None, rgi=None):
    if os.path.exists(folder_mosaic + f'mosaic_RGI_{rgi}.tif'):
        dem_rgi = rioxarray.open_rasterio(folder_mosaic + f'mosaic_RGI_{rgi}.tif')
    else:  # We have to create the mosaic
        print(f"Mosaic {rgi} not present. Let's create it on the fly.")
        dem_rgi = create_mosaic_rgi_tandemx(rgi=rgi, path_rgi_tiles=folder_mosaic, save=0)
    return dem_rgi

def find_tandemx_tiles(minx, miny, maxx, maxy, rgi, path_tandemx):
    """Find either the tile or the tandemx tiles that contain the glacier
    See https://geoservice.dlr.de/web/dataguide/tdm30/"""
    if isinstance(rgi, int):
        rgi = f"{rgi:02d}"

    folder_rgi_tiles = f"{path_tandemx}RGI_{rgi}"
    print(folder_rgi_tiles)

    # Calculate the latitude and longitude values needed for tandemx lookup
    tile_minx, tile_miny = int(minx), int(miny)
    tile_maxx, tile_maxy = int(maxx), int(maxy)
    print(minx, miny, maxx, maxy)
    print((tile_minx, tile_miny), (tile_maxx, tile_maxy))

    def get_NS(x):
        if x>=0: return 'N'
        else: return 'S'
    def get_EW(x):
        if x>=0: return 'E'
        else: return 'W'

    # Now I have to look for the tile(s) that contain (tile_minx, tile_miny) and (tile_maxx, tile_maxy)
    #all_files = sorted(glob.glob(f"{folder_rgi_tiles}/*", recursive = False))
    code1 = f"{get_NS(tile_miny)}{tile_miny:02d}{get_EW(tile_minx)}{tile_minx:03d}"
    code2 = f"{get_NS(tile_maxy)}{tile_maxy:02d}{get_EW(tile_maxx)}{tile_maxx:03d}"
    dif_lat = maxy - miny
    dif_lon = maxx - minx

    list_tiles = []
    tile1 = sorted(glob.glob(f"{folder_rgi_tiles}/TDM1_EDEM_*_{code1}_V01_C", recursive = False))
    tile2 = sorted(glob.glob(f"{folder_rgi_tiles}/TDM1_EDEM_10_{code2}_V01_C", recursive = False))
    list_tiles.extend(tile1)
    list_tiles.extend(tile2)

    print(code1, 'Tile:', tile1)
    print(code2, 'Tile:', tile2)
    print(f"List of tiles I need to import: {list_tiles}")

    mosaic = None
    return mosaic

def create_mosaic_rgi_tandemx(rgi=None, path_rgi_tiles=None, save=0):

    if isinstance(rgi, int):
        rgi = f"{rgi:02d}"

    tqdm.write(f"Begin creation of mosaic for region {rgi}...")
    folder_rgi = f"{path_rgi_tiles}RGI_{rgi}"

    src_files_to_mosaic = []
    list_rgi_w84tiles = glob.glob(f"{folder_rgi}/*/EDEM/*_W84.tif", recursive = False)
    tqdm.write(f"In rgi {rgi} we have {len(list_rgi_w84tiles)} tiles to be merged")

    for i, filename in tqdm(enumerate(list_rgi_w84tiles), total=len(list_rgi_w84tiles), desc=f"rgi {rgi} importing tiles", leave=False):

        #tqdm.write(f"rgi:{rgi}, import tile {i+1}/{len(list_rgi_w84tiles)}")
        src = rioxarray.open_rasterio(list_rgi_w84tiles[i], cache=False)
        src.rio.write_crs("EPSG:4326", inplace=True)
        src = src.where(src != src.rio.nodata) # replace nodata (-32767).0 with nans.
        src.rio.write_nodata(np.nan, inplace=True)  # set nodata as nan
        src_files_to_mosaic.append(src)

    # See https://geoservice.dlr.de/web/dataguide/tdm30/pdfs/TD_GS_PS_0215_TanDEM_X_30m_Edited_DEM_Product_Description_v1_1.pdf
    # Since lon resolution changes, I'd like to force the mosaic to have a arcsec resolution in both lat, lon.
    # The lon res can be e.g. 5/3600 and I'd like that to be 1/3600. Probably rioxarray merging operates some interpolation.
    # We extract here 1/3600 resolution. And specify such value when merging below.
    res_out = min(np.abs(src.rio.resolution()))
    assert res_out == 1./3600, "Unexpected resolution for merging tiles."


    mosaic_rgi = merge_arrays(src_files_to_mosaic, res=(res_out, res_out), nodata=np.nan)
    tqdm.write(f"Mosaic for rgi {rgi} created.")
    #print('Resolution mosaic:', mosaic_rgi.rio.resolution())

    plot_mosaic = False
    if plot_mosaic:
        mosaic_clipped = mosaic_rgi.rio.clip_box(
            minx=-95,
            miny=80,
            maxx=-90,
            maxy=80.5,
        )
        fig, ax1 = plt.subplots()
        im1 = mosaic_clipped.plot(ax=ax1, cmap='terrain')
        plt.show()


    # Save
    if save:
        tqdm.write(f"Saving mosaic ...")
        mosaic_rgi.rio.to_raster(f"{path_rgi_tiles}mosaic_RGI_{rgi}.tif") # I could compress='deflate'/...
        tqdm.write(f"mosaic_RGI_{rgi}.tif saved")

    return mosaic_rgi


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create DEM mosaic from TandemX-EDEM tiles')
    parser.add_argument("--input", type=str, default="/media/nico/samsung_nvme/Tandem-X-EDEM/",
                        help="folder path to the TandemX-EDEM tiles")
    parser.add_argument("--region", type=int, default=None, help="RGI region in x format")
    parser.add_argument("--save", type=int, default=0, help="Save mosaic: 0/1")

    args = parser.parse_args()

    mosaic_rgi = create_mosaic_rgi_tandemx(rgi=args.region, path_rgi_tiles=args.input, save=args.save)

