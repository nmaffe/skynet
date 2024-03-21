import os, glob
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd
import oggm
from oggm import utils
import matplotlib
import matplotlib.pyplot as plt
from create_rgi_mosaic_tanxedem import get_NS, get_EW, min_closest_multiple, max_closest_multiple, getTDXlonres, get_codes

parser = argparse.ArgumentParser()
parser.add_argument('--outfolder', type=str, default="/media/nico/samsung_nvme/Tandem-X-EDEM/", help="Outpath for saved files")
parser.add_argument('--rgi', type=int, default=8, help="Region to produce list of url")
parser.add_argument('--buffer', type=float, default=1./8, help="Buffer to glacier extent")
parser.add_argument('--save', type=int, default=0, help="Save it or go home")

args = parser.parse_args()

utils.get_rgi_dir(version='62')

rgi = args.rgi
buffer = args.buffer

list_rgi_tiles = []
list_rgi_tiles_url = []

gdf = gpd.read_file(utils.get_rgi_region_file(f"{rgi:02d}", version='62'))
fig, ax = plt.subplots()

for idx, row in gdf.iterrows():
    glacier = gdf.iloc[idx]
    geometry = glacier['geometry']
    ax.plot(*geometry.exterior.xy, c='red')

    minx, miny, maxx, maxy = geometry.bounds
    # Add some buffer
    minx = minx - buffer
    miny = miny - buffer
    maxx = maxx + buffer
    maxy = maxy + buffer
    #print(miny, minx, maxy, maxx)

    codes_tiles_for_glacier = get_codes(miny, minx, maxy, maxx)
    list_rgi_tiles.extend(codes_tiles_for_glacier)

# We now have a list of tile code names.
list_rgi_tiles  = list(set(list_rgi_tiles))  # Remove duplicates
print(f"In rgi {rgi} we have produced {len(list_rgi_tiles)} tiles.")

# We now need to format them as Tandem-X wants the names for download:
#https://download.geoservice.dlr.de/TDM30_EDEM/files/TDM1_EDEM_10_N45/TDM1_EDEM_10_N45E010/TDM1_EDEM_10_N45E013_V01_C/TDM1_EDEM_10_N45E013_V01_C.zip
for code in list_rgi_tiles:
    NSlat = code[0:3]
    EWlon = code[3:7]
    EWlon_mul10 = EWlon[:-1] + '0'

    # Create url and append to list
    file_code = f"https://download.geoservice.dlr.de/TDM30_EDEM/files/TDM1_EDEM_10_{NSlat}/TDM1_EDEM_10_{NSlat}{EWlon_mul10}" \
                f"/TDM1_EDEM_10_{NSlat}{EWlon}_V01_C/TDM1_EDEM_10_{NSlat}{EWlon}_V01_C.zip"
    list_rgi_tiles_url.append(file_code)
    #print(code, NSlat, EWlon, EWlon_mul10, file_code)

print(f"Finished rgi {rgi}.")
#for i in list_rgi_tiles_url: print(i)

# Save to txt
if args.save:
    print(f"Saved file for rgi {rgi}")
    np.savetxt(f'{args.outfolder}TDM30_EDEM-url-list-rgi-{args.rgi}.txt', list_rgi_tiles_url, fmt='%s')

# Extract the maximum x value from the limits
plot=True
if plot:
    min_lon, max_lon = min(ax.get_xlim()), max(ax.get_xlim())
    min_lon, max_lon = int(min_lon), int(np.ceil(max_lon))
    min_lat, max_lat = min(ax.get_ylim()), max(ax.get_ylim())
    min_lat, max_lat = int(min_lat), int(np.ceil(max_lat))
    #print(min_lon, max_lon)
    #print(min_lat, max_lat)
    ax.vlines(np.arange(min_lon, max_lon), ymin=min_lat, ymax=max_lat, color='grey', linestyle='-', alpha=0.5)
    ax.hlines(np.arange(min_lat, max_lat), xmin=min_lon, xmax=max_lon, color='grey', linestyle='-', alpha=0.5)
    plt.xticks(range(min_lon, max_lon + 1))
    # Set y-labels to multiples of 1
    plt.yticks(range(min_lat, max_lat + 1))
    plt.show()


