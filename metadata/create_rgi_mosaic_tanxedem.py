import os, glob
import argparse
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rioxarray.merge import merge_arrays
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Create DEM mosaic from TandemX-EDEM tiles')
parser.add_argument("--input",  type=str, default="/media/nico/samsung_nvme/Tandem-X-EDEM",
                    help="folder path to the TandemX-EDEM tiles")
parser.add_argument("--region",  type=int, default=None, help="RGI region in x format")
parser.add_argument("--save",  type=bool, default=False, help="Save mosaic")

args = parser.parse_args()

rgi = args.region
folder_rgi = f"{args.input}/RGI_{rgi:02d}"

src_files_to_mosaic = []
list_rgi_w84tiles = glob.glob(f"{folder_rgi}/*/EDEM/*_W84.tif", recursive = False)
print(f"In rgi {rgi} we have {len(list_rgi_w84tiles)} tiles to be merged")

for i, filename in enumerate(list_rgi_w84tiles):

    print(f"rgi:{rgi}, import tile {i+1}/{len(list_rgi_w84tiles)}")

    src = rioxarray.open_rasterio(list_rgi_w84tiles[i])
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

print(f"Begin creation of mosaic..")
mosaic_rgi = merge_arrays(src_files_to_mosaic, res=(res_out, res_out), nodata=np.nan)
print(f"Mosaic for rgi {rgi} done.")
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
if args.save:
    mosaic_rgi.rio.to_raster(f"{args.input}/mosaic_tdx_RGI_{rgi:02d}.tif")
    print(f"mosaic_tdx_RGI_{rgi:02d}.tif saved")
