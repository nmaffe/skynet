import os
import sys
sys.path.append("/home/nico/PycharmProjects/skynet/code") # to import haversine from utils.py
from utils import haversine
from glob import glob
import argparse
import numpy as np
import xarray
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray
import geopandas as gpd
import oggm
from oggm import utils
from shapely.geometry import Point, Polygon
from math import radians, cos, sin, asin, sqrt

"""
The purpose of this program is to produce the glacier metadata 
at for some random locations inside the glacier geometry. 

Input: glacier name (RGIId), how many points you want to generate. 
Output: pandas dataframe with features calculated for each generated point. 
"""
# todo: check that the masks I am using to do the inpainting already account for the nunataks. That's important.
# todo also check the same is done for create_metadata in calculating the distance from border
# todo (=nunatak if that is the case)
parser = argparse.ArgumentParser()
parser.add_argument('--mosaic', type=str,default="/media/nico/samsung_nvme/ASTERDEM_v3_mosaics/",
                    help="Path to DEM mosaics")

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


def create_random_lon_lat_points_inside_glacier(glacier_name, n=50):
    print(f"Investigating glacier {glacier_name}")

    rgi = int(glacier_name[6:8]) # get rgi from the glacier code
    oggm_rgi_shp = utils.get_rgi_region_file(rgi, version='62') # get rgi region shp
    oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)             # get rgi dataset of glaciers

    # Get glacier dataset and necessary stuff
    try:
        gl_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name]
        print(gl_df.T)
        print(f"Glacier {glacier_name} found.")
    except Exception as e: print("Error {e}")
    assert len(gl_df) == 1, "Check this please."
    # print(gl_df)
    gl_geom = gl_df['geometry'].item() # glacier geometry Polygon
    gl_geom_ext = Polygon(gl_geom.exterior)  # glacier geometry Polygon
    gl_geom_nunataks_list = [Polygon(nunatak) for nunatak in gl_geom.interiors] # list of nunataks Polygons
    llx, lly, urx, ury = gl_geom.bounds # geometry bounds

    # Dictionary of points to be generated
    points = {'lons':[], 'lats':[], 'nunataks':[]}
    points_df = pd.DataFrame(columns=CFG.features+['lons', 'lats', 'nunataks'])

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
            if is_nunatak:
                points['nunataks'].append(1)
            else:
                points['nunataks'].append(0)


    # Fill lats, lons and nunataks
    points_df['lats']=points['lats']
    points_df['lons']=points['lons']
    points_df['nunataks']=points['nunataks']

    # Let's start filling the other features
    points_df['RGI'] = rgi
    points_df['Area'] = gl_df['Area'].item()
    points_df['Zmin'] = gl_df['Zmin'].item()
    points_df['Zmax'] = gl_df['Zmax'].item()
    points_df['Zmed'] = gl_df['Zmed'].item()
    points_df['Slope'] = gl_df['Slope'].item()
    points_df['Lmax'] = gl_df['Lmax'].item()

    """ Add Slopes and Elevation """
    dem_rgi = rioxarray.open_rasterio(args.mosaic + f'mosaic_RGI_{rgi:02d}.tif')
    ris_ang = dem_rgi.rio.resolution()[0]

    swlat = points_df['lats'].min()
    swlon = points_df['lons'].min()
    nelat = points_df['lats'].max()
    nelon = points_df['lons'].max()
    deltalat = np.abs(swlat - nelat)
    deltalon = np.abs(swlon - nelon)
    lats_xar = xarray.DataArray(points_df['lats'])
    lons_xar = xarray.DataArray(points_df['lons'])

    eps = 5 * ris_ang

    # clip
    try:
        focus = dem_rgi.rio.clip_box(
            minx=swlon - (deltalon + eps),
            miny=swlat - (deltalat + eps),
            maxx=nelon + (deltalon + eps),
            maxy=nelat + (deltalat + eps)
        )
    except:
        input(f"Problemi")

    focus = focus.squeeze()

    lon_c = (0.5 * (focus.coords['x'][-1] + focus.coords['x'][0])).to_numpy()
    lat_c = (0.5 * (focus.coords['y'][-1] + focus.coords['y'][0])).to_numpy()
    ris_metre_lon = haversine(lon_c, lat_c, lon_c + ris_ang, lat_c) * 1000
    ris_metre_lat = haversine(lon_c, lat_c, lon_c, lat_c + ris_ang) * 1000

    # calculate slope for restricted dem
    dz_dlat, dz_dlon = np.gradient(focus.values, ris_metre_lat, ris_metre_lon)  # [m/m]

    dz_dlat_xarray = focus.copy(data=dz_dlat)
    dz_dlon_xarray = focus.copy(data=dz_dlon)

    # interpolate slope
    slope_lat_data = dz_dlat_xarray.interp(y=lats_xar, x=lons_xar, method='linear').data # (N,)
    slope_lon_data = dz_dlon_xarray.interp(y=lats_xar, x=lons_xar, method='linear').data # (N,)

    # interpolate dem
    elevation_data = focus.interp(y=lats_xar, x=lons_xar, method='linear').data # (N,)

    assert slope_lat_data.shape == slope_lon_data.shape == elevation_data.shape, "Different shapes, something wrong!"

    # Fill dataframe with slope_lat, slope_lon, slope, elevation_astergdem and elevation_from_zmin
    points_df['slope_lat'] = slope_lat_data
    points_df['slope_lon'] = slope_lon_data
    points_df['elevation_astergdem'] = elevation_data
    points_df['elevation_from_zmin'] = points_df['elevation_astergdem'] - points_df['Zmin']
    points_df['slope'] = np.sqrt(points_df['slope_lat'] ** 2 + points_df['slope_lon'] ** 2)

    print(points_df.T)

    # Show the result
    show_glacier_with_produced_points = True
    if show_glacier_with_produced_points:
        fig, axes = plt.subplots(1,2, figsize=(8,4))
        ax1, ax2 = axes.flatten()

        for ax in axes:
            ax.plot(*gl_geom_ext.exterior.xy, lw=1, c='red')
            for interior in gl_geom.interiors:
                ax.plot(*interior.xy, lw=1, c='blue')
            for (lon, lat, nunatak) in zip(points['lons'], points['lats'], points['nunataks']):
                if nunatak: ax.scatter(lon, lat, s=50, lw=2, c='b', zorder=1)
                else: ax.scatter(lon, lat, s=50, lw=2, c='r', ec='r', zorder=1)

        im1 = dz_dlat_xarray.plot(ax=ax1, cmap='gist_gray', vmin=np.nanmin(slope_lat_data),
                                  vmax=np.nanmax(slope_lat_data), zorder=0)
        s1 = ax1.scatter(x=lons_xar, y=lats_xar, s=50, c=slope_lat_data, ec=None, cmap='gist_gray',
                         vmin=np.nanmin(slope_lat_data), vmax=np.nanmax(slope_lat_data), zorder=1)

        im2 = focus.plot(ax=ax2, cmap='gist_gray', vmin=np.nanmin(elevation_data),
                                  vmax=np.nanmax(elevation_data), zorder=0)
        s2 = ax2.scatter(x=lons_xar, y=lats_xar, s=50, c=elevation_data, ec=None, cmap='gist_gray',
                         vmin=np.nanmin(elevation_data), vmax=np.nanmax(elevation_data), zorder=1)


        plt.show()
    return points_df

df_points = create_random_lon_lat_points_inside_glacier(glacier_name='RGI60-11.01450', n=25)
#glacier_name =  'RGI60-11.01450' Aletsch
#glacier_name = np.random.choice(RGI_burned)
