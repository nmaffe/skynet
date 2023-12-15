import os
from glob import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import oggm
from oggm import utils
from shapely.geometry import Point, Polygon

"""
The purpose of this program is to fetch the glacier metadata 
at random locations and produce ice thickness predictions at these locations 
using the the GNN.
Input: glacier geometry, GNN trained model
Output: some kind of array of ice thickness predictions that can be used by the inpainting model.  
"""
# todo: check that the masks I am using to do the inpainting already account for the nunataks. That's important.
# todo also check the same is done for create_metadata in calculating the distance from border
# todo (=nunatak if that is the case)
parser = argparse.ArgumentParser()
parser.add_argument('--OGGM_folder', type=str,default="/home/nico/OGGM", help="Path to OGGM main folder")

args = parser.parse_args()

utils.get_rgi_dir(version='62')  # setup oggm version

class CFG:
    # I need to reconstruct these features for each point created inside the glacier polygon
    features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
     'dist_from_border_km', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
     'elevation_from_zmin', 'RGI']
RGI_burned = ['RGI60-11.00562', 'RGI60-11.00590', 'RGI60-11.00603', 'RGI60-11.00638', 'RGI60-11.00647',
                  'RGI60-11.00689', 'RGI60-11.00695', 'RGI60-11.00846', 'RGI60-11.00950', 'RGI60-11.01024',
                  'RGI60-11.01041', 'RGI60-11.01067', 'RGI60-11.01144', 'RGI60-11.01199', 'RGI60-11.01296',
                  'RGI60-11.01344', 'RGI60-11.01367', 'RGI60-11.01376', 'RGI60-11.01473', 'RGI60-11.01509',
                  'RGI60-11.01576', 'RGI60-11.01604', 'RGI60-11.01698', 'RGI60-11.01776', 'RGI60-11.01786',
                  'RGI60-11.01790', 'RGI60-11.01791', 'RGI60-11.01806', 'RGI60-11.01813', 'RGI60-11.01840',
                  'RGI60-11.01857', 'RGI60-11.01894', 'RGI60-11.01928', 'RGI60-11.01962', 'RGI60-11.01986',
                  'RGI60-11.02006', 'RGI60-11.02024', 'RGI60-11.02027', 'RGI60-11.02244', 'RGI60-11.02249',
                  'RGI60-11.02261', 'RGI60-11.02448', 'RGI60-11.02490', 'RGI60-11.02507', 'RGI60-11.02549',
                  'RGI60-11.02558', 'RGI60-11.02583', 'RGI60-11.02584', 'RGI60-11.02596', 'RGI60-11.02600',
                  'RGI60-11.02624', 'RGI60-11.02673', 'RGI60-11.02679', 'RGI60-11.02704', 'RGI60-11.02709',
                  'RGI60-11.02715', 'RGI60-11.02740', 'RGI60-11.02745', 'RGI60-11.02755', 'RGI60-11.02774',
                  'RGI60-11.02775', 'RGI60-11.02787', 'RGI60-11.02796', 'RGI60-11.02864', 'RGI60-11.02884',
                  'RGI60-11.02890', 'RGI60-11.02909', 'RGI60-11.03249']

gl_id = 'RGI60-11.00590' # 'RGI60-11.01450' Aletsch
gl_id = np.random.choice(RGI_burned)
print(f"Glacier {gl_id}")

def create_random_lon_lat_points_inside_glacier(glname, n=50):

    rgi = int(glname[6:8]) # get rgi from the glacier code
    oggm_rgi_shp = utils.get_rgi_region_file(rgi, version='62') # get rgi region shp
    oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)             # get rgi dataset of glaciers

    # Get glacier dataset and necessary stuff
    try: gl_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glname]
    except Exception as e: print("Error {e}")
    assert len(gl_df) == 1, "Check this please."
    # print(gl_df)
    gl_geom = gl_df['geometry'].item() # glacier geometry Polygon
    gl_geom_ext = Polygon(gl_geom.exterior)  # glacier geometry Polygon
    gl_geom_nunataks_list = [Polygon(nunatak) for nunatak in gl_geom.interiors] # list of nunataks Polygons
    llx, lly, urx, ury = gl_geom.bounds # geometry bounds

    # Dictionary of points to be generated
    points = {'lons':[], 'lats':[], 'nunataks':[]}

    while (len(points['lons']) < n):
        r_lon = np.random.uniform(llx, urx)
        r_lat = np.random.uniform(lly, ury)
        point = Point(r_lon, r_lat)

        is_inside = gl_geom_ext.contains(point)
        if is_inside:
            points['lons'].append(r_lon)
            points['lats'].append(r_lat)

            # Flag as 1 if point inside any nunatak.
            # If glacier does not contain nunataks, the list will be empty and 0 will populate automatically.
            is_nunatak = any(nunatak.contains(point) for nunatak in gl_geom_nunataks_list)
            if is_nunatak: points['nunataks'].append(1)
            else: points['nunataks'].append(0)

    # Show the result
    show_glacier_with_produced_points = True
    if show_glacier_with_produced_points:
        plt.plot(*gl_geom_ext.exterior.xy, c='red')
        for interior in gl_geom.interiors:
            plt.plot(*interior.xy, c='blue')
        for (lon, lat, nunatak) in zip(points['lons'], points['lats'], points['nunataks']):
            if nunatak: plt.scatter(lon, lat, s=20, c='b')
            else: plt.scatter(lon, lat, s=20, c='r')
        plt.show()

    return points

dict_points = create_random_lon_lat_points_inside_glacier(gl_id, n=10)

#todo: now I have to populate the dictionary with metadata. Effectively I have to calculate the metadata features
# for each (lon, lat) point the I generated. Follow create_metadata.py for guidance.
