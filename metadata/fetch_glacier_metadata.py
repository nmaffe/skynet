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
from rioxarray import merge
import geopandas as gpd
import oggm
from oggm import utils
from shapely.geometry import Point, Polygon
from pyproj import Proj, transform, Transformer, CRS
from math import radians, cos, sin, asin, sqrt, floor
import utm
import shapely.wkt


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
parser.add_argument('--millan_velocity_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/velocity/",
                    help="Path to Millan velocity data")
parser.add_argument('--millan_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/thickness/",
                    help="Path to Millan ice thickness data")


args = parser.parse_args()

utils.get_rgi_dir(version='62')  # setup oggm version

class CFG:
    # I need to reconstruct these features for each point created inside the glacier polygon
    features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
     'dist_from_border_km', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
     'elevation_from_zmin', 'RGI', 'ith_m']

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


def from_lat_lon_to_utm_and_epsg(lat, lon):
    """https://github.com/Turbo87/utm"""
    # Note lat lon can be also NumPy arrays.
    # In this case zone letter and number will be calculate from first entry.
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    southern_hemisphere_TrueFalse = True if zone_letter < 'N' else False
    epsg_code = 32600 + zone_number + southern_hemisphere_TrueFalse * 100
    return (easting, northing, zone_number, zone_letter, epsg_code)


def populate_glacier_with_metadata(glacier_name, n=50):
    print(f"Investigating glacier {glacier_name}")

    rgi = int(glacier_name[6:8]) # get rgi from the glacier code
    oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62') # get rgi region shp
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

    # Generate points
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

    """ Calculate Millan vx, vy, v """
    # get Millan vx files and create mosaic
    files_vx = glob(args.millan_velocity_folder + 'RGI-{}/VX_RGI-{}*'.format(rgi, rgi))
    xds_4326 = []
    for tiffile in files_vx:
        xds = rioxarray.open_rasterio(tiffile, masked=False)
        xds.rio.write_nodata(np.nan, inplace=True)
        xds_4326.append(xds)
    mosaic_vx = merge.merge_arrays(xds_4326)
    files_vy = glob(args.millan_velocity_folder + 'RGI-{}/VY_RGI-{}*'.format(rgi, rgi))
    # get Millan vy files and create mosaic
    xds_4326 = []
    for tiffile in files_vy:
        xds = rioxarray.open_rasterio(tiffile, masked=False)
        xds.rio.write_nodata(np.nan, inplace=True)
        xds_4326.append(xds)
    mosaic_vy = merge.merge_arrays(xds_4326)
    # get Millan ith files and create mosaic
    files_ith = glob(args.millan_icethickness_folder + 'RGI-{}/THICKNESS_RGI-{}*'.format(rgi, rgi))
    xds_4326 = []
    for tiffile in files_ith:
        xds = rioxarray.open_rasterio(tiffile, masked=False)
        xds.rio.write_nodata(np.nan, inplace=True)
        xds_4326.append(xds)
    mosaic_ith = merge.merge_arrays(xds_4326)

    bounds_ith = mosaic_ith.rio.bounds()
    bounds_vx = mosaic_vx.rio.bounds()
    bounds_vy = mosaic_vy.rio.bounds()

    # Reshape the 3 mosaic if different shapes
    if (bounds_ith != bounds_vx or bounds_ith != bounds_vy or bounds_vx != bounds_vy):
        new_llx = max(bounds_ith[0], bounds_vx[0], bounds_vy[0])
        new_lly = max(bounds_ith[1], bounds_vx[1], bounds_vy[1])
        new_urx = min(bounds_ith[2], bounds_vx[2], bounds_vy[2])
        new_ury = min(bounds_ith[3], bounds_vx[3], bounds_vy[3])
        mosaic_ith = mosaic_ith.rio.clip_box(minx=new_llx, miny=new_lly, maxx=new_urx, maxy=new_ury)
        mosaic_vx = mosaic_vx.rio.clip_box(minx=new_llx, miny=new_lly, maxx=new_urx, maxy=new_ury)
        mosaic_vy = mosaic_vy.rio.clip_box(minx=new_llx, miny=new_lly, maxx=new_urx, maxy=new_ury)
        print(f"Reshaped Millan bounds.")

    # Mask ice thickness based on velocity map
    mosaic_ith = mosaic_ith.where(mosaic_vx.notnull())

    # Note ris_ang for lat/lon/ith mosaics may be slightly different even for the same rgi, but we don't care here
    ris_ang_millan = mosaic_vx.rio.resolution()[0]
    crs = mosaic_vx.rio.crs
    eps_millan = 10 * ris_ang_millan
    transformer = Transformer.from_crs("EPSG:4326", crs)

    # Covert lat lon coordinates to Millan projection
    lons_crs, lats_crs = transformer.transform(points_df['lats'].to_numpy(), points_df['lons'].to_numpy())

    # clip millan mosaic around the generated points
    try:
        focus_vx = mosaic_vx.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                          maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan)
        focus_vy = mosaic_vy.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                          maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan)
        focus_ith = mosaic_ith.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                          maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan)

    except:
        print(f"No Millan data for rgi {rgi} glacier {glacier_name}")
        # todo: what to do if vx, vy, ith are not defined for this glacier ? How do I fill the dataframe ?
        # In principle I can still define all other variables except for v and maybe distance_from_border

    # Interpolate
    # Note that since Millan products although are defined to rgi6, points generated very very close
    # to the glaciers boundaries may result in nan interpolation. This should be removed at the end.
    vx_data = focus_vx.interp(y=xarray.DataArray(lats_crs), x=xarray.DataArray(lons_crs), method="linear").data.squeeze()
    vy_data = focus_vy.interp(y=xarray.DataArray(lats_crs), x=xarray.DataArray(lons_crs), method="linear").data.squeeze()
    ith_data = focus_ith.interp(y=xarray.DataArray(lats_crs), x=xarray.DataArray(lons_crs), method="linear").data.squeeze()
    print(f"From Millan vx, vy, ith we have generated {np.isnan(vx_data).sum()}/{np.isnan(vy_data).sum()}/{np.isnan(ith_data).sum()} nans.")

    # Fill dataframe with vx, vy, ith_m
    points_df['vx'] = vx_data # note this may contain nans
    points_df['vy'] = vy_data # note this may contain nans
    points_df['ith_m'] = ith_data # note this may contain nans
    points_df['v'] = np.sqrt(points_df['vx']**2 + points_df['vy']**2) # note this may contain nans

    """ Calculate distance_from_border """
    # Now we know all points are generated inside the geometry. Do we still to compute the distance from the
    # edge using Millan maps ? Can we use a different approach ?
    for (lon, lat, nunatak) in zip(points_df['lons'], points_df['lats'], points_df['nunataks']):
        point = Point(lon, lat)
        print(point)

        easting, nothing, zonenum, zonelett, epsg = from_lat_lon_to_utm_and_epsg(lat, lon)
        print(easting, nothing, zonenum, zonelett, epsg)

        """Using pyproj
        original_projection = CRS("EPSG:4326")  # WGS 84 (latitude, longitude)
        target_projection = CRS(f"EPSG:{epsg}")  # World Mercator (meters)
        transformer = Transformer.from_crs(original_projection, target_projection, always_xy=False)

        # Transform the coordinates of the geometry
        coordinates_transformed = transformer.transform(*zip(*gl_geom_ext.exterior.coords))
        # Create a new Shapely Polygon with the transformed coordinates
        geom_transformed = Polygon(zip(*coordinates_transformed))
        # distance
        distance = point.distance(geom_transformed)
        print('Distance:', distance)"""

        """Using geopandas"""
        #print(len(gl_df['geometry'].item()))
        #sth = gpd.GeoDataFrame(geometry=[shapely.wkt.loads(f"{gl_df['geometry'].item()}")], crs="EPSG:4326")


        point_single = gpd.GeoDataFrame(geometry=[shapely.wkt.loads(f"POINT ({lon} {lat})")], crs="EPSG:4326")
        line = gpd.GeoDataFrame(geometry=[shapely.wkt.loads(f"{gl_geom}")], crs="EPSG:4326")
        lines_nunataks = gpd.GeoDataFrame(geometry=[shapely.wkt.loads(f"{Polygon(nunatak)}") for nunatak in gl_geom.interiors], crs="EPSG:4326")

        line_to_epsg = line.to_crs(epsg=epsg)
        lines_nunataks_to_epsg = lines_nunataks.to_crs(epsg=epsg)
        point_to_epsg = point_single.to_crs(epsg=epsg)
        print(point_to_epsg)

        distance_from_periphery = line_to_epsg.exterior.distance(point_to_epsg)
        distance_from_nunataks = 0#lines_nunataks_to_epsg.exterior.distance(point_to_epsg)

        print('Distance from periphery:', distance_from_periphery)
        print('Distance from nunataks:', distance_from_nunataks)


        """ Here's the winner. """
        exterior_ring = gl_geom.exterior # shapely.geometry.polygon.LinearRing
        interior_rings = gl_geom.interiors # shapely.geometry.polygon.InteriorRingSequence
        #for interior in interior_rings: print(type(interior)) # shapely.geometry.polygon.LinearRing
        all_rings_list = gpd.GeoSeries([exterior_ring] + list(interior_rings), crs="EPSG:4326") # this list will contain all geometries
        all_points_list = gpd.GeoSeries([point] * len(all_rings_list), crs="EPSG:4326")
        all_rings_list_epsg = all_rings_list.to_crs(epsg=epsg)
        all_point_list_epsg = all_points_list.to_crs(epsg=epsg)
        distance_between_geoseries = all_point_list_epsg.distance(all_rings_list_epsg)
        print(distance_between_geoseries)
        input('wait02')



        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.plot(*gl_geom_ext.exterior.xy, lw=1, c='red')
        for interior in gl_geom.interiors:
            ax1.plot(*interior.xy, lw=1, c='blue')
        if nunatak: ax1.scatter(lon, lat, s=50, lw=2, c='b')
        else: ax1.scatter(lon, lat, s=50, lw=2, c='r', ec='r')

        ax2.plot(*line_to_epsg.loc[0, 'geometry'].exterior.xy, lw=1, c='red')
        ax2.scatter(*point_to_epsg.loc[0, 'geometry'].xy, s=50, lw=2, c='r', ec='r')
        plt.show()
        input('wait')

        #distance = point.distance(gl_geom_ext)
        #print(f"The distance between the point and the polygon is: {distance}")




    # Show the result
    show_glacier_with_produced_points = False
    if show_glacier_with_produced_points:
        fig, axes = plt.subplots(1,3, figsize=(8,4))
        ax1, ax2, ax3 = axes.flatten()

        for ax in (ax1, ax2):
            ax.plot(*gl_geom_ext.exterior.xy, lw=1, c='red')
            for interior in gl_geom.interiors:
                ax.plot(*interior.xy, lw=1, c='blue')
            for (lon, lat, nunatak) in zip(points['lons'], points['lats'], points['nunataks']):
                if nunatak: ax.scatter(lon, lat, s=50, lw=2, c='b', zorder=1)
                else: ax.scatter(lon, lat, s=50, lw=2, c='r', ec='r', zorder=1)

        # slope_lat
        im1 = dz_dlat_xarray.plot(ax=ax1, cmap='gist_gray', vmin=np.nanmin(slope_lat_data),
                                  vmax=np.nanmax(slope_lat_data), zorder=0)
        s1 = ax1.scatter(x=lons_xar, y=lats_xar, s=50, c=slope_lat_data, ec=None, cmap='gist_gray',
                         vmin=np.nanmin(slope_lat_data), vmax=np.nanmax(slope_lat_data), zorder=1)

        # elevation
        im2 = focus.plot(ax=ax2, cmap='gist_gray', vmin=np.nanmin(elevation_data),
                                  vmax=np.nanmax(elevation_data), zorder=0)
        s2 = ax2.scatter(x=lons_xar, y=lats_xar, s=50, c=elevation_data, ec=None, cmap='gist_gray',
                         vmin=np.nanmin(elevation_data), vmax=np.nanmax(elevation_data), zorder=1)

        # vx
        im3 = focus_vx.plot(ax=ax3, cmap='viridis', vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data))
        s3 = ax3.scatter(x=lons_crs, y=lats_crs, s=50, c=vx_data, ec=(1, 0, 0, 1), cmap='viridis',
                         vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data), zorder=1)
        s3_1 = ax3.scatter(x=lons_crs[np.argwhere(np.isnan(vx_data))], y=lats_crs[np.argwhere(np.isnan(vx_data))], s=50,
                           c='magenta', zorder=1)


        plt.show()

    # todo: remove any nan line (can be in vx, vy, v, ith_m)
    #print(points_df.T)
    return points_df

df_points = populate_glacier_with_metadata(glacier_name='RGI60-11.01450', n=1) #RGI60-08.00001
#glacier_name =  'RGI60-11.01450' Aletsch # RGI60-11.02774
#glacier_name = np.random.choice(RGI_burned)
