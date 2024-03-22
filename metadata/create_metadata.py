import os, sys, time
import pandas as pd
from glob import glob
import random
import xarray, rioxarray, rasterio
import xrspatial.curvature
import xrspatial.aspect
import argparse
from rioxarray import merge
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import oggm
from oggm import cfg, utils, workflow, tasks, graphics
import geopandas as gpd
from tqdm import tqdm
import scipy
from scipy import spatial
from sklearn.neighbors import KDTree
import shapely
from shapely.geometry import box, Point, Polygon, LinearRing, LineString, MultiLineString
from matplotlib.colors import Normalize, LogNorm
from shapely.ops import unary_union, nearest_points
from pyproj import Transformer, CRS, Geod
import utm
from joblib import Parallel, delayed
from create_rgi_mosaic_tanxedem import create_mosaic_rgi_tandemx, create_glacier_tile_dem_mosaic
from utils_metadata import from_lat_lon_to_utm_and_epsg, gaussian_filter_with_nans, haversine

"""
This program creates a dataframe of metadata for the points in glathida.

Run time evaluated on on rgi = [1,3,4,7,8,11,18]: 9.6h
Note: rgi 6 has no glathida data.

1. add_rgi. Time: 2min.
2. add_RGIId_and_OGGM_stats. Time: 10 min (+rgi 5 in 14min)
3. add_slopes_elevation. Time: 80 min. (+rgi 5 in 32min)
    - No nan can be produced here. 
4. add_millan_vx_vy_ith. Time: 4.5 h (rgi1 30m, rgi3 3h, rgi4 40m, rgi7 35m)
    - Points inside the glacier but close to the borders can be interpolated as nan.
    - Note: method to interpolate is chosen as "nearest" to reduce as much as possible these nans.
5. add_dist_from_boder_using_geometries. 1.5h (rgi1 30m, rgi3 30m, rgi4 13m, rgi7 17m) (+rgi5: 1.5h)
6. add_farinotti_ith. Time: 2h (rgi1 1h, rgi3 0.5h, rgi4 10m, rgi7 8m, rgi8 3m, rgi11 10m, rgi18 3m) (+rgi 5 in 35m)
    - Points inside the glacier but close to the borders can be interpolated as nan.
    - Note: method to interpolate is chosen as "nearest" to reduce as much as possible these nans.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--path_ttt_csv', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT.csv",
                    help="Path to TTT.csv file")
parser.add_argument('--path_ttt_rgi_csv', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_rgi.csv",
                    help="Path to TTT_rgi.csv file")
parser.add_argument('--path_O1Regions_shp', type=str,default="/home/nico/OGGM/rgi/RGIV62/00_rgi62_regions/00_rgi62_O1Regions.shp",
                    help="Path to OGGM's 00_rgi62_O1Regions.shp shapefiles of all 19 RGI regions")
parser.add_argument('--mosaic', type=str,default="/media/nico/samsung_nvme/Tandem-X-EDEM/",
                    help="Path to DEM mosaics")
parser.add_argument('--millan_velocity_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/velocity/",
                    help="Path to Millan velocity data")
parser.add_argument('--millan_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/thickness/",
                    help="Path to Millan ice thickness data")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Farinotti/",
                    help="Path to Farinotti ice thickness data")
parser.add_argument('--OGGM_folder', type=str,default="/home/nico/OGGM", help="Path to OGGM main folder")
parser.add_argument('--save', type=int, default=0, help="Save final dataset or not.")
parser.add_argument('--save_outname', type=str,
            default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/metadata6.csv",
            help="Saved dataframe name.")

#todo: whenever i call clip_box i need to check if there is only 1 measurement !
#todo: change all reprojecting method from nearest to something else, like bisampling. Also in all .to_crs(...)
# todo: Slope from oggm has some strangely high values (or is it expressed in %?). Worth thicking of calculating it myself ?

# todo:
#  00) Add adaptive filter to Millan velocity. I THINK DONE
#  0) Slope/elevation function: avoid the mosaic thing and only collect the necessary tiles. DONE
#  1) Add rgi 5. Why in rgi 5 glaciers should be 19306 and instead oggm lists them as 20261 ?
#  I believe OGGM glaciers are slightly different from the official RGI, they have their V62 version.
#  2) Bonus: add feature slope interpolation at closest point.

# TODO: IMPORTANT !!! VV AS PROVIDED FROM SAT PRODUCT SHOULD BE USED INSTEAD OF MY MANUAL CALCULATION.
#  ALSO CHECK IF THE UNIT OF THE VELOCITY MATCH WITH THE UNIT OF THE UNDERLYING GRID SIZE. THIS IS IMPORTANT WHEN
#  I THEN DIFFERENTIATE. IF [VX]=M/YR AND I DIFFERENTIATE IN UTM WITH GRID SPACING IN METERS, IT SHOULD BE FINE.
#  I FEEL I MAY NEED TO IMPORT V RATHER THAN CALCULATING IT MYSELF.



utils.get_rgi_dir(version='62')
utils.get_rgi_intersects_dir(version='62')

args = parser.parse_args()

""" Add rgi values """
def add_rgi(glathida, path_O1_shp):
    print(f'Adding RGI method...')

    if ('RGI' in list(glathida)):
        print('Variable RGI already in dataframe.')
        return glathida

    world = gpd.read_file(path_O1_shp)
    glathida['RGI'] = [np.nan]*len(glathida)
    lats = glathida['POINT_LAT']
    lons = glathida['POINT_LON']
    points = [Point(ilon, ilat) for (ilon, ilat) in zip(lons, lats)]

    # Define the regions
    region1a = world.loc[0]['geometry']
    region1b = world.loc[1]['geometry']
    region1 = shapely.ops.unary_union([region1a, region1b]) #class 'shapely.geometry.polygon.Polygon
    region2 = world.loc[2]['geometry']
    region3 = world.loc[3]['geometry']
    region4 = world.loc[4]['geometry']
    region5 = world.loc[5]['geometry']
    region6 = world.loc[6]['geometry']
    region7 = world.loc[7]['geometry']
    region8 = world.loc[8]['geometry']
    region9 = world.loc[9]['geometry']
    region10a = world.loc[10]['geometry']
    region10b = world.loc[11]['geometry']
    region10 = shapely.ops.unary_union([region10a, region10b]) # shapely.geometry.multipolygon.MultiPolygon
    region11 = world.loc[12]['geometry']
    region12 = world.loc[13]['geometry']
    region13 = world.loc[14]['geometry']
    region14 = world.loc[15]['geometry']
    region15 = world.loc[16]['geometry']
    region16 = world.loc[17]['geometry']
    region17 = world.loc[18]['geometry']
    region18 = world.loc[19]['geometry']
    region19 = world.loc[20]['geometry']

    all_regions = [region1, region2, region3, region4, region5, region6, region7, region8, region9, region10,
                   region11, region12, region13, region14, region15, region16, region17, region18, region19]

    # loop over the region geometries
    for n, region in tqdm(enumerate(all_regions), total=len(all_regions), leave=True):

        # mask True/False to decide whether the points are inside the region geometry
        mask_points_in_region = region.contains(points)

        # select only those points inside the glacier
        df_poins_in_region = glathida[mask_points_in_region]

        # add to dataframe
        glathida.loc[df_poins_in_region.index, 'RGI'] = n+1

    print(glathida['RGI'].value_counts())
    print(glathida['RGI'].count()) # does not count nans

    ifplot = False
    if ifplot:
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        num_colors = len(all_regions)
        bounds = np.linspace(1, num_colors, num_colors)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # rgi boundaries colors
        colors = cmap(norm(bounds))

        fig, ax1 = plt.subplots()
        ax1.scatter(lons, lats, s=1, c=glathida['RGI'], cmap=cmap, norm=norm)

        for n, rgi_geom in enumerate(all_regions):
            if (rgi_geom.geom_type == 'Polygon'):
                ax1.plot(*rgi_geom.exterior.xy, c=colors[n])
            elif (rgi_geom.geom_type == 'MultiPolygon'):
                for geom in rgi_geom.geoms:
                    ax1.plot(*geom.exterior.xy, c=colors[n])
            else: raise ValueError("Geom type not recognized. Check.")
        plt.show()

    return glathida

""" Add Slopes and Elevation """
def add_slopes_elevation(glathida, path_mosaic):
    print('Running slope and elevation method...')
    ts = time.time()

    if (any(ele in list(glathida) for ele in ['slope_lat', 'slope_lon', 'elevation'])):
        print('Variables slope_lat, slope_lon or elevation already in dataframe.')
        return glathida

    glathida['elevation'] = [np.nan] * len(glathida)
    glathida['slope_lat'] = [np.nan] * len(glathida)
    glathida['slope_lon'] = [np.nan] * len(glathida)
    glathida['slope_lat_gf50'] = [np.nan] * len(glathida)
    glathida['slope_lon_gf50'] = [np.nan] * len(glathida)
    glathida['slope_lat_gf100'] = [np.nan] * len(glathida)
    glathida['slope_lon_gf100'] = [np.nan] * len(glathida)
    glathida['slope_lat_gf150'] = [np.nan] * len(glathida)
    glathida['slope_lon_gf150'] = [np.nan] * len(glathida)
    glathida['slope_lat_gf300'] = [np.nan] * len(glathida)
    glathida['slope_lon_gf300'] = [np.nan] * len(glathida)
    glathida['slope_lat_gf450'] = [np.nan] * len(glathida)
    glathida['slope_lon_gf450'] = [np.nan] * len(glathida)
    glathida['slope_lat_gfa'] = [np.nan] * len(glathida)
    glathida['slope_lon_gfa'] = [np.nan] * len(glathida)
    glathida['curv_50'] = [np.nan] * len(glathida)
    glathida['curv_300'] = [np.nan] * len(glathida)
    glathida['curv_gfa'] = [np.nan] * len(glathida)
    glathida['aspect_50'] = [np.nan] * len(glathida)
    glathida['aspect_300'] = [np.nan] * len(glathida)
    glathida['aspect_gfa'] = [np.nan] * len(glathida)
    datax = []  # just to analyse the results
    datay = []  # just to analyse the results

    for rgi in [1,3,4,7,8,11,18]:

        # get dem o create it
        #if os.path.exists(path_mosaic + f'mosaic_RGI_{rgi:02d}.tif'):
        #    dem_rgi = rioxarray.open_rasterio(path_mosaic + f'mosaic_RGI_{rgi:02d}.tif')
        #    print(f"mosaic_RGI_{rgi:02d}.tif found")
        #else:  # We have to create the mosaic
        #    print(f"Mosaic {rgi} not present. Let's create it on the fly...")
        #    dem_rgi = create_mosaic_rgi_tandemx(rgi=rgi, path_rgi_tiles=path_mosaic, save=0)

        #ris_ang = dem_rgi.rio.resolution()[0]
        # eps = 5 * ris_ang

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi] # collapse glathida to specific rgi
        ids_rgi = glathida_rgi['GlaThiDa_ID'].unique().tolist()  # unique IDs

        for id_rgi in tqdm(ids_rgi, total=len(ids_rgi), desc=f"rgi {rgi} Glathida ID", leave=True):

            # todo: appears to be a problem for rgi 5 id_rgi 2092, 2102 ? Maybe not..

            glathida_id = glathida_rgi.loc[glathida_rgi['GlaThiDa_ID'] == id_rgi]  # collapse glathida_rgi to specific id
            glathida_id = glathida_id.copy()

            glathida_id['northings'] = np.nan
            glathida_id['eastings'] = np.nan
            glathida_id['epsg'] = np.nan
            for idx in glathida_id.index:
                lat = glathida_id.at[idx, 'POINT_LAT']
                lon = glathida_id.at[idx, 'POINT_LON']
                e, n, _, _, epsg = from_lat_lon_to_utm_and_epsg(lat, lon)
                glathida_id.at[idx, 'northings'] = n
                glathida_id.at[idx, 'eastings'] = e
                glathida_id.at[idx, 'epsg'] = int(epsg)

            # get unique epsgs
            unique_epsgs = glathida_id['epsg'].unique().astype(int).tolist()

            for epsg_unique in unique_epsgs:
                glathida_id_epsg = glathida_id.loc[glathida_id['epsg'] == epsg_unique]
                indexes_all_epsg = glathida_id_epsg.index.tolist()

                lats = np.array(glathida_id_epsg['POINT_LAT'])
                lons = np.array(glathida_id_epsg['POINT_LON'])
                northings = np.array(glathida_id_epsg['northings'])
                eastings = np.array(glathida_id_epsg['eastings'])

                swlat = np.amin(lats)
                swlon = np.amin(lons)
                nelat = np.amax(lats)
                nelon = np.amax(lons)

                deltalat = np.abs(swlat - nelat)
                deltalon = np.abs(swlon - nelon)
                delta = max(deltalat, deltalon, 0.1)
                northings_xar = xarray.DataArray(northings)
                eastings_xar = xarray.DataArray(eastings)


                # clip
                try:
                    #focus = dem_rgi.rio.clip_box(
                    #    minx = swlon - delta,
                    #    miny = swlat - delta,
                    #    maxx = nelon + delta,
                    #    maxy = nelat + delta,
                    #)
                    focus = create_glacier_tile_dem_mosaic(minx=swlon - delta,
                                                            miny=swlat - delta,
                                                            maxx=nelon + delta,
                                                            maxy=nelat + delta,
                                                            rgi=rgi, path_tandemx=path_mosaic)
                except:
                    raise ValueError(f"Problems in method add_slopes_elevation for rgi {rgi} glacier_id: {id_rgi}, "
                                     f"glacier box {swlon - delta} {swlat - delta} {nelon + delta} {nelat + delta}")

                focus = focus.squeeze()

                # Reproject to utm (projection distortions along boundaries converted to nans)
                focus_utm = focus.rio.reproject(epsg_unique, resampling=rasterio.enums.Resampling.bilinear, nodata=-9999)
                focus_utm = focus_utm.where(focus_utm != -9999, np.nan)

                # clip the utm with a buffer of 2 km in both dimentions
                focus_utm_clipped = focus_utm.rio.clip_box(
                    minx=min(eastings)-2000,
                    miny=min(northings)-2000,
                    maxx=max(eastings)+2000,
                    maxy=max(northings)+2000)

                # Calculate the resolution in meters of the utm focus (resolutions in x and y are the same!)
                res_utm_metres = focus_utm_clipped.rio.resolution()[0]

                # Calculate sigma in meters for adaptive gaussian fiter
                sigma_af_min, sigma_af_max = 100.0, 2000.0
                try:
                    # Each id_rgi may come with multiple area values and also nans (probably if all points outside glacier geometries)
                    area_id = glathida_id_epsg['Area'].min()  # km2
                    lmax_id = glathida_id_epsg['Lmax'].max()  # m
                    a = 1e6*area_id/(np.pi*0.5*lmax_id)
                    sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
                    #print(area_id, lmax_id, a, value)
                except Exception as e:
                    sigma_af = sigma_af_min
                # Ensure that our value correctly in range [50.0, 2000.0]
                assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"
                #print(f"Adaptive gaussian filter with sigma = {value} meters.")

                # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
                num_px_sigma_50 = max(1, round(50/res_utm_metres))
                num_px_sigma_100 = max(1, round(100/res_utm_metres))
                num_px_sigma_150 = max(1, round(150/res_utm_metres))
                num_px_sigma_300 = max(1, round(300/res_utm_metres))
                num_px_sigma_450 = max(1, round(450/res_utm_metres))
                num_px_sigma_af = max(1, round(sigma_af / res_utm_metres))

                # Apply filter
                focus_filter_50_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_50, trunc=4.0)
                focus_filter_100_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_100, trunc=4.0)
                focus_filter_150_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_150, trunc=4.0)
                focus_filter_300_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_300, trunc=4.0)
                focus_filter_450_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_450, trunc=4.0)
                focus_filter_af_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_af, trunc=3.0)

                # Mask back the filtered arrays
                focus_filter_50_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_50_utm)
                focus_filter_100_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_100_utm)
                focus_filter_150_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_150_utm)
                focus_filter_300_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_300_utm)
                focus_filter_450_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_450_utm)
                focus_filter_af_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_af_utm)

                # create xarray object of filtered dem
                focus_filter_xarray_50_utm = focus_utm_clipped.copy(data=focus_filter_50_utm)
                focus_filter_xarray_100_utm = focus_utm_clipped.copy(data=focus_filter_100_utm)
                focus_filter_xarray_150_utm = focus_utm_clipped.copy(data=focus_filter_150_utm)
                focus_filter_xarray_300_utm = focus_utm_clipped.copy(data=focus_filter_300_utm)
                focus_filter_xarray_450_utm = focus_utm_clipped.copy(data=focus_filter_450_utm)
                focus_filter_xarray_af_utm = focus_utm_clipped.copy(data=focus_filter_af_utm)

                # calculate slopes for restricted dem
                # using numpy.gradient dz_dlat, dz_dlon = np.gradient(focus_utm_clipped.values, -res_utm_metres, res_utm_metres)  # [m/m]
                dz_dlat_xar, dz_dlon_xar = focus_utm_clipped.differentiate(coord='y'), focus_utm_clipped.differentiate(coord='x')
                dz_dlat_filter_xar_50, dz_dlon_filter_xar_50 = focus_filter_xarray_50_utm.differentiate(coord='y'), focus_filter_xarray_50_utm.differentiate(coord='x')
                dz_dlat_filter_xar_100, dz_dlon_filter_xar_100 = focus_filter_xarray_100_utm.differentiate(coord='y'), focus_filter_xarray_100_utm.differentiate(coord='x')
                dz_dlat_filter_xar_150, dz_dlon_filter_xar_150 = focus_filter_xarray_150_utm.differentiate(coord='y'), focus_filter_xarray_150_utm.differentiate(coord='x')
                dz_dlat_filter_xar_300, dz_dlon_filter_xar_300  = focus_filter_xarray_300_utm.differentiate(coord='y'), focus_filter_xarray_300_utm.differentiate(coord='x')
                dz_dlat_filter_xar_450, dz_dlon_filter_xar_450  = focus_filter_xarray_450_utm.differentiate(coord='y'), focus_filter_xarray_450_utm.differentiate(coord='x')
                dz_dlat_filter_xar_af, dz_dlon_filter_xar_af  = focus_filter_xarray_af_utm.differentiate(coord='y'), focus_filter_xarray_af_utm.differentiate(coord='x')

                # Calculate curvature and aspect using xrspatial
                # Units of the curvature output (1/100) of a z-unit. Units of aspect are between [0, 360]
                # Note that xrspatial using a standard 3x3 grid around pixel to calculate stuff
                # Note that xrspatial produces nans at boundaries, but that should not be a problem for interpolation.
                curv_50 = xrspatial.curvature(focus_filter_xarray_50_utm)
                curv_300 = xrspatial.curvature(focus_filter_xarray_300_utm)
                curv_af = xrspatial.curvature(focus_filter_xarray_af_utm)
                aspect_50 = xrspatial.aspect(focus_filter_xarray_50_utm)
                aspect_300 = xrspatial.aspect(focus_filter_xarray_300_utm)
                aspect_af = xrspatial.aspect(focus_filter_xarray_af_utm)

                # interpolate slope and dem
                elevation_data = focus_utm_clipped.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lat_data = dz_dlat_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lon_data = dz_dlon_xar.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lat_data_filter_50 = dz_dlat_filter_xar_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lon_data_filter_50 = dz_dlon_filter_xar_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lat_data_filter_100 = dz_dlat_filter_xar_100.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lon_data_filter_100 = dz_dlon_filter_xar_100.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lat_data_filter_150 = dz_dlat_filter_xar_150.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lon_data_filter_150 = dz_dlon_filter_xar_150.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lat_data_filter_300 = dz_dlat_filter_xar_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lon_data_filter_300 = dz_dlon_filter_xar_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lat_data_filter_450 = dz_dlat_filter_xar_450.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lon_data_filter_450 = dz_dlon_filter_xar_450.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lat_data_filter_af = dz_dlat_filter_xar_af.interp(y=northings_xar, x=eastings_xar, method='linear').data
                slope_lon_data_filter_af = dz_dlon_filter_xar_af.interp(y=northings_xar, x=eastings_xar, method='linear').data
                curv_data_50 = curv_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                curv_data_300 = curv_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                curv_data_af = curv_af.interp(y=northings_xar, x=eastings_xar, method='linear').data
                aspect_data_50 = aspect_50.interp(y=northings_xar, x=eastings_xar, method='linear').data
                aspect_data_300 = aspect_300.interp(y=northings_xar, x=eastings_xar, method='linear').data
                aspect_data_af = aspect_af.interp(y=northings_xar, x=eastings_xar, method='linear').data

                # check if any nan in the interpolate data
                contains_nan = any(np.isnan(arr).any() for arr in [slope_lon_data, slope_lat_data,
                                                                   slope_lon_data_filter_50, slope_lat_data_filter_50,
                                                                   slope_lon_data_filter_100, slope_lat_data_filter_100,
                                                                   slope_lon_data_filter_150, slope_lat_data_filter_150,
                                                                   slope_lon_data_filter_300, slope_lat_data_filter_300,
                                                                   slope_lon_data_filter_450, slope_lat_data_filter_450,
                                                                   slope_lon_data_filter_af, slope_lat_data_filter_af,
                                                                   curv_data_50, curv_data_300, curv_data_af,
                                                                   aspect_data_50, aspect_data_300, aspect_data_af])
                if contains_nan:
                    raise ValueError(f"Nan detected in elevation/slope calc. Check")

                # other checks
                assert slope_lat_data.shape == slope_lon_data.shape == elevation_data.shape, "Different shapes, something wrong!"
                assert slope_lat_data_filter_150.shape == slope_lon_data_filter_150.shape == elevation_data.shape, "Different shapes, something wrong!"
                assert len(slope_lat_data) == len(indexes_all_epsg), "Different shapes, something wrong!"
                assert curv_data_50.shape == curv_data_300.shape == curv_data_af.shape, "Different shapes, something wrong!"
                assert aspect_data_50.shape == aspect_data_300.shape == aspect_data_af.shape, "Different shapes, something wrong!"

                # write to dataframe
                glathida.loc[indexes_all_epsg, 'elevation'] = elevation_data
                glathida.loc[indexes_all_epsg, 'slope_lat'] = slope_lat_data
                glathida.loc[indexes_all_epsg, 'slope_lon'] = slope_lon_data
                glathida.loc[indexes_all_epsg, 'slope_lat_gf50'] = slope_lat_data_filter_50
                glathida.loc[indexes_all_epsg, 'slope_lon_gf50'] = slope_lon_data_filter_50
                glathida.loc[indexes_all_epsg, 'slope_lat_gf100'] = slope_lat_data_filter_100
                glathida.loc[indexes_all_epsg, 'slope_lon_gf100'] = slope_lon_data_filter_100
                glathida.loc[indexes_all_epsg, 'slope_lat_gf150'] = slope_lat_data_filter_150
                glathida.loc[indexes_all_epsg, 'slope_lon_gf150'] = slope_lon_data_filter_150
                glathida.loc[indexes_all_epsg, 'slope_lat_gf300'] = slope_lat_data_filter_300
                glathida.loc[indexes_all_epsg, 'slope_lon_gf300'] = slope_lon_data_filter_300
                glathida.loc[indexes_all_epsg, 'slope_lat_gf450'] = slope_lat_data_filter_450
                glathida.loc[indexes_all_epsg, 'slope_lon_gf450'] = slope_lon_data_filter_450
                glathida.loc[indexes_all_epsg, 'slope_lat_gfa'] = slope_lat_data_filter_af
                glathida.loc[indexes_all_epsg, 'slope_lon_gfa'] = slope_lon_data_filter_af
                glathida.loc[indexes_all_epsg, 'curv_50'] = curv_data_50
                glathida.loc[indexes_all_epsg, 'curv_300'] = curv_data_300
                glathida.loc[indexes_all_epsg, 'curv_gfa'] = curv_data_af
                glathida.loc[indexes_all_epsg, 'aspect_50'] = aspect_data_50
                glathida.loc[indexes_all_epsg, 'aspect_300'] = aspect_data_300
                glathida.loc[indexes_all_epsg, 'aspect_gfa'] = aspect_data_af

                plot_curvature = False
                if plot_curvature:
                    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
                    im1 = focus_filter_xarray_50_utm.plot(ax=ax1, cmap='viridis')
                    im2 = focus_filter_xarray_300_utm.plot(ax=ax2, cmap='viridis')
                    im3 = curv_50.plot(ax=ax3, cmap='viridis')
                    im4 = curv_300.plot(ax=ax4, cmap='viridis')
                    im5 = aspect_50.plot(ax=ax5, cmap='viridis')
                    im6 = aspect_300.plot(ax=ax6, cmap='viridis')
                    plt.show()


                plot_utm = False
                if plot_utm:
                    fig, ((ax0, ax1, ax2, ax3), (ax01, ax4, ax5, ax6)) = plt.subplots(2, 4, figsize=(8, 5))

                    im0 =focus.plot(ax=ax0, cmap='viridis', )

                    im01 =focus_utm.plot(ax=ax01, cmap='viridis', )
                    s01 = ax01.scatter(x=eastings, y=northings, s=15, c='k')

                    im1 = focus_utm_clipped.plot(ax=ax1, cmap='viridis', vmin=focus_utm_clipped.min(),vmax=focus_utm_clipped.max())
                    s1 = ax1.scatter(x=eastings, y=northings, s=15, c=elevation_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                     vmin=focus_utm_clipped.min(), vmax=focus_utm_clipped.max(), zorder=1)

                    im2 = dz_dlat_xar.plot(ax=ax2, cmap='viridis', vmin=dz_dlat_xar.min(), vmax=dz_dlat_xar.max())
                    s2 = ax2.scatter(x=eastings, y=northings, s=15, c=slope_lat_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                     vmin=dz_dlat_xar.min(), vmax=dz_dlat_xar.max(), zorder=1)

                    im3 = dz_dlon_xar.plot(ax=ax3, cmap='viridis', vmin=dz_dlon_xar.min(), vmax=dz_dlon_xar.max())
                    s3 = ax3.scatter(x=eastings, y=northings, s=15, c=slope_lon_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                     vmin=dz_dlon_xar.min(), vmax=dz_dlon_xar.max(), zorder=1)

                    im4 = focus_filter_xarray_300_utm.plot(ax=ax4, cmap='viridis', )
                    s4 = ax4.scatter(x=eastings, y=northings, s=15, c='k')

                    im5 = dz_dlat_filter_xar_300.plot(ax=ax5, cmap='viridis', vmin=dz_dlat_filter_xar_300.min(), vmax=dz_dlat_filter_xar_300.max())
                    s5 = ax5.scatter(x=eastings, y=northings, s=15, c=slope_lat_data_filter_300, ec=(0, 0, 0, 0.1),
                                     cmap='viridis', vmin=dz_dlat_filter_xar_300.min(), vmax=dz_dlat_filter_xar_300.max(), zorder=1)
                    im6 = dz_dlon_filter_xar_300.plot(ax=ax6, cmap='viridis', vmin=dz_dlon_filter_xar_300.min(), vmax=dz_dlon_filter_xar_300.max())
                    s6 = ax6.scatter(x=eastings, y=northings, s=15, c=slope_lon_data_filter_300, ec=(0, 0, 0, 0.1),
                                     cmap='viridis', vmin=dz_dlon_filter_xar_300.min(), vmax=dz_dlon_filter_xar_300.max(), zorder=1)

                    plt.show()


            """
            # I want such amount of pixels to have an equivalent sigma of 50, 100, 150, 300m.
            sigma_50_lon, sigma_50_lat = max(1, round(50./ris_metre_lon)), max(1, round(50./ris_metre_lat))
            sigma_100_lon, sigma_100_lat = max(1, round(100./ris_metre_lon)), max(1, round(100./ris_metre_lat))
            sigma_150_lon, sigma_150_lat = max(1, round(150./ris_metre_lon)), max(1, round(150./ris_metre_lat))
            sigma_300_lon, sigma_300_lat = max(1, round(300./ris_metre_lon)), max(1, round(300./ris_metre_lat))

            # Apply filter to the restricted dem
            focus_filter_50 = scipy.ndimage.gaussian_filter(focus.values, sigma=[sigma_50_lat, sigma_50_lon], mode='nearest')
            focus_filter_100 = scipy.ndimage.gaussian_filter(focus.values, sigma=[sigma_100_lat, sigma_100_lon], mode='nearest')
            focus_filter_150 = scipy.ndimage.gaussian_filter(focus.values, sigma=[sigma_150_lat, sigma_150_lon], mode='nearest')
            focus_filter_300 = scipy.ndimage.gaussian_filter(focus.values, sigma=[sigma_300_lat, sigma_300_lon], mode='nearest')
            # create xarray object of filtered dem
            focus_filter_xarray_50 = focus.copy(data=focus_filter_50)
            focus_filter_xarray_100 = focus.copy(data=focus_filter_100)
            focus_filter_xarray_150 = focus.copy(data=focus_filter_150)
            focus_filter_xarray_300 = focus.copy(data=focus_filter_300)

            # calculate slopes for restricted dem
            dz_dlat, dz_dlon = np.gradient(focus.values, -ris_metre_lat, ris_metre_lon)  # [m/m]
            dz_dlat_filter_50, dz_dlon_filter_50 = np.gradient(focus_filter_50, -ris_metre_lat, ris_metre_lon)  # [m/m]
            dz_dlat_filter_100, dz_dlon_filter_100 = np.gradient(focus_filter_100, -ris_metre_lat, ris_metre_lon)  # [m/m]
            dz_dlat_filter_150, dz_dlon_filter_150 = np.gradient(focus_filter_150, -ris_metre_lat, ris_metre_lon)  # [m/m]
            dz_dlat_filter_300, dz_dlon_filter_300 = np.gradient(focus_filter_300, -ris_metre_lat, ris_metre_lon)  # [m/m]

            # create xarray object of slopes
            dz_dlat_xar, dz_dlon_xar = focus.copy(data=dz_dlat), focus.copy(data=dz_dlon)
            dz_dlat_filter_xar_50, dz_dlon_filter_xar_50 = focus.copy(data=dz_dlat_filter_50), focus.copy(data=dz_dlon_filter_50)
            dz_dlat_filter_xar_100, dz_dlon_filter_xar_100 = focus.copy(data=dz_dlat_filter_100), focus.copy(data=dz_dlon_filter_100)
            dz_dlat_filter_xar_150, dz_dlon_filter_xar_150 = focus.copy(data=dz_dlat_filter_150), focus.copy(data=dz_dlon_filter_150)
            dz_dlat_filter_xar_300, dz_dlon_filter_xar_300 = focus.copy(data=dz_dlat_filter_300), focus.copy(data=dz_dlon_filter_300)

            # interpolate slope and dem
            slope_lat_data = dz_dlat_xar.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lon_data = dz_dlon_xar.interp(y=lats_xar, x=lons_xar, method='linear').data
            elevation_data = focus.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lat_data_filter_50 = dz_dlat_filter_xar_50.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lon_data_filter_50 = dz_dlon_filter_xar_50.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lat_data_filter_100 = dz_dlat_filter_xar_100.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lon_data_filter_100 = dz_dlon_filter_xar_100.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lat_data_filter_150 = dz_dlat_filter_xar_150.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lon_data_filter_150 = dz_dlon_filter_xar_150.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lat_data_filter_300 = dz_dlat_filter_xar_300.interp(y=lats_xar, x=lons_xar, method='linear').data
            slope_lon_data_filter_300 = dz_dlon_filter_xar_300.interp(y=lats_xar, x=lons_xar, method='linear').data

            if (len(slope_lat_data.shape) != 0):
                datax.extend(list(slope_lat_data))
                datay.extend(list(slope_lon_data))
            else:
                datax.append(slope_lat_data.item())
                datay.append(slope_lon_data.item())

            ifplot = False
            #if (np.any(np.isnan(slope_lat_data)) or np.any(np.isnan(slope_lon_data))): ifplot = True
            #if (np.any(slope_lat_data>3)): ifplot = True
            if ifplot:
                fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(8, 5))

                im1 = focus.plot(ax=ax1, cmap='viridis', )

                im2 = dz_dlat_xar.plot(ax=ax2, cmap='viridis', vmin=np.nanmin(slope_lat_data), vmax=np.nanmax(slope_lat_data))
                s2 = ax2.scatter(x=lons, y=lats, s=15, c=slope_lat_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                 vmin=np.nanmin(slope_lat_data), vmax=np.nanmax(slope_lat_data), zorder=1)
                s2_1 = ax2.scatter(x=lons[np.argwhere(np.isnan(slope_lat_data))],
                                   y=lats[np.argwhere(np.isnan(slope_lat_data))], s=15, c='magenta', zorder=1)
                im3 = dz_dlon_xar.plot(ax=ax3, cmap='viridis', vmin=np.nanmin(slope_lon_data), vmax=np.nanmax(slope_lon_data))
                s3 = ax3.scatter(x=lons, y=lats, s=15, c=slope_lon_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                 vmin=np.nanmin(slope_lon_data), vmax=np.nanmax(slope_lon_data), zorder=1)
                s3_1 = ax3.scatter(x=lons[np.argwhere(np.isnan(slope_lon_data))],
                                   y=lats[np.argwhere(np.isnan(slope_lon_data))], s=15, c='magenta', zorder=1)

                im4 = focus_filter_xarray_300.plot(ax=ax4, cmap='viridis',)
                im5 = dz_dlat_filter_xar_300.plot(ax=ax5, cmap='viridis', vmin=np.nanmin(slope_lat_data_filter_300), vmax=np.nanmax(slope_lat_data_filter_300))
                s5 = ax5.scatter(x=lons, y=lats, s=15, c=slope_lat_data_filter_300, ec=(0, 0, 0, 0.1), cmap='viridis',
                                 vmin=np.nanmin(slope_lat_data_filter_300), vmax=np.nanmax(slope_lat_data_filter_300), zorder=1)
                s5_1 = ax5.scatter(x=lons[np.argwhere(np.isnan(slope_lat_data_filter_300))],
                                   y=lats[np.argwhere(np.isnan(slope_lat_data_filter_300))], s=15, c='magenta', zorder=1)
                im6 = dz_dlon_filter_xar_300.plot(ax=ax6, cmap='viridis', vmin=np.nanmin(slope_lon_data_filter_300), vmax=np.nanmax(slope_lon_data_filter_300))
                s6 = ax6.scatter(x=lons, y=lats, s=15, c=slope_lon_data_filter_300, ec=(0, 0, 0, 0.1), cmap='viridis',
                                 vmin=np.nanmin(slope_lon_data_filter_300), vmax=np.nanmax(slope_lon_data_filter_300), zorder=1)
                s6_1 = ax6.scatter(x=lons[np.argwhere(np.isnan(slope_lon_data_filter_300))],
                                   y=lats[np.argwhere(np.isnan(slope_lon_data_filter_300))], s=15, c='magenta', zorder=1)

                plt.tight_layout()
                plt.show()

            assert slope_lat_data.shape == slope_lon_data.shape == elevation_data.shape, "Different shapes, something wrong!"
            assert slope_lat_data_filter_150.shape == slope_lon_data_filter_150.shape == elevation_data.shape, "Different shapes, something wrong!"

            # write to dataframe
            glathida.loc[indexes_all, 'slope_lat'] = slope_lat_data
            glathida.loc[indexes_all, 'slope_lon'] = slope_lon_data
            glathida.loc[indexes_all, 'elevation'] = elevation_data
            glathida.loc[indexes_all, 'slope_lat_gf50'] = slope_lat_data_filter_50
            glathida.loc[indexes_all, 'slope_lon_gf50'] = slope_lon_data_filter_50
            glathida.loc[indexes_all, 'slope_lat_gf100'] = slope_lat_data_filter_100
            glathida.loc[indexes_all, 'slope_lon_gf100'] = slope_lon_data_filter_100
            glathida.loc[indexes_all, 'slope_lat_gf150'] = slope_lat_data_filter_150
            glathida.loc[indexes_all, 'slope_lon_gf150'] = slope_lon_data_filter_150
            glathida.loc[indexes_all, 'slope_lat_gf300'] = slope_lat_data_filter_300
            glathida.loc[indexes_all, 'slope_lon_gf300'] = slope_lon_data_filter_300
            """
    print(f"Slopes and elevation done in {(time.time()-ts)/60} min")
    return glathida

"""Add Millan's velocity vx, vy, ith"""
def add_millan_vx_vy_ith(glathida, path_millan_velocity, path_millan_icethickness):
    """
    :param glathida: input csv. dataframe to which I want to add velocity
    :param path_millan_velocity: path to folder with millan velocity data
    :param path_millan_icethickness: path to folder with millan ice thickness data
    :return: dataframe with velocity added
    See https://www.sedoo.fr/theia-publication-products/?uuid=55acbdd5-3982-4eac-89b2-46703557938c for info.
    """
    print('Adding Millan velocity and ice thickness method...')
    tm = time.time()

    if (any(ele in list(glathida) for ele in ['vx', 'vy', 'ith_m'])):
        print('Variable already in dataframe.')
        #return glathida

    glathida['ith_m'] = [np.nan] * len(glathida)
    glathida['vx'] = [np.nan] * len(glathida)
    glathida['vy'] = [np.nan] * len(glathida)
    glathida['vx_gf50'] = [np.nan] * len(glathida)
    glathida['vx_gf100'] = [np.nan] * len(glathida)
    glathida['vx_gf150'] = [np.nan] * len(glathida)
    glathida['vx_gf300'] = [np.nan] * len(glathida)
    glathida['vx_gf450'] = [np.nan] * len(glathida)
    glathida['vx_gfa'] = [np.nan] * len(glathida)
    glathida['vy_gf50'] = [np.nan] * len(glathida)
    glathida['vy_gf100'] = [np.nan] * len(glathida)
    glathida['vy_gf150'] = [np.nan] * len(glathida)
    glathida['vy_gf300'] = [np.nan] * len(glathida)
    glathida['vy_gf450'] = [np.nan] * len(glathida)
    glathida['vy_gfa'] = [np.nan] * len(glathida)
    glathida['dvx_dx'] = [np.nan] * len(glathida)
    glathida['dvx_dy'] = [np.nan] * len(glathida)
    glathida['dvy_dx'] = [np.nan] * len(glathida)
    glathida['dvy_dy'] = [np.nan] * len(glathida)

    for rgi in [5]:#[1,3,4,7,8,11,18]:
        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]  # glathida to specific rgi
        tqdm.write(f'rgi: {rgi}, Total points: {len(glathida_rgi)}')

        files_vx = sorted(glob(f"{path_millan_velocity}RGI-{rgi}/VX_RGI-{rgi}*"))
        files_vy = sorted(glob(f"{path_millan_velocity}RGI-{rgi}/VY_RGI-{rgi}*"))
        files_ith = sorted(glob(f"{path_millan_icethickness}RGI-{rgi}/THICKNESS_RGI-{rgi}*"))
        n_rgi_tiles = len(files_vx)

        for i, (file_vx, file_vy, file_ith) in enumerate(zip(files_vx, files_vy, files_ith)):

            tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
            tile_vy = rioxarray.open_rasterio(file_vy, masked=False)
            tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

            """Rasterio detects the raster as in EPSG:32629. Millan says another thing (3414). Who should I believe ?
            How does rasterio detects the EPSG, can he be wrong ? Is the file wrong ? Is the file correct but the EPSG is wrong ?
            I should try reproject in 4326 and see how it looks. 
            ith files are: EPSG:3413 Sea Ice Polar Stereographic North
            Ok Millan mentions velocity should be this one, which does not fully cover the entire greenland
            https://nsidc.org/data/nsidc-0646/versions/3
            However, there is also this one from insar (should be better?) https://nsidc.org/data/nsidc-0478/versions/2
            https://nsidc.org/data/nsidc-0481/versions/4
            The spatial coverage however is still a mistery
            https://nsidc.org/data/nsidc-0670/versions/1#anchor-1
            https://nsidc.org/data/nsidc-0725/versions/5"""
            if not file_ith=="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/thickness/RGI-5/THICKNESS_RGI-5.2_2022February24.tif":
                continue
            print(file_vx, tile_vx.rio.crs)
            print(file_vy, tile_vy.rio.crs)
            print(file_ith, tile_ith.rio.crs)
            print(tile_vx)
            tile_ith.rio.reproject("EPSG:4326").plot(cmap='viridis')
            plt.show()
            exit('wait')

            tile_vx.rio.write_nodata(np.nan, inplace=True)
            tile_vy.rio.write_nodata(np.nan, inplace=True)
            tile_ith.rio.write_nodata(np.nan, inplace=True)

            assert tile_vx.rio.crs == tile_vy.rio.crs == tile_ith.rio.crs, "Tiles vx, vy, ith with different epsg."
            assert tile_vx.rio.bounds() == tile_vy.rio.bounds(), "Tiles vx, vy bounds not the same. Super strange."
            assert tile_vx.rio.resolution() == tile_vy.rio.resolution() == tile_ith.rio.resolution(), \
                "Tiles vx, vy, ith have different resolution."

            # vx (and vy) and ith may have a bound difference for some reason (ask Millan).
            # In such cases we reindex the ith tile to match the vx tile coordinates
            if not tile_vx.rio.bounds() == tile_ith.rio.bounds():
                tile_ith = tile_ith.reindex_like(tile_vx, method="nearest", tolerance=50., fill_value=np.nan)
            assert tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_vx.rio.bounds(), "All tiles bounds not the same"

            # Mask ice thickness based on velocity map
            tile_ith = tile_ith.where(tile_vx.notnull())

            # Convert tile boundaries to lat lon
            tran_to_4326 = Transformer.from_crs(tile_vx.rio.crs, "EPSG:4326")
            lat0, lon0 = tran_to_4326.transform(tile_vx.rio.bounds()[0], tile_vx.rio.bounds()[1])
            lat1, lon1 = tran_to_4326.transform(tile_vx.rio.bounds()[2], tile_vx.rio.bounds()[3])
            #print(f"Tile {i, file_vx} bounds {lat0, lon0, lat1, lon1}")

            # Select data inside the tile
            condition1 = glathida_rgi['POINT_LAT']>lat0
            condition2 = glathida_rgi['POINT_LAT']<lat1
            condition3 = glathida_rgi['POINT_LON']>lon0
            condition4 = glathida_rgi['POINT_LON']<lon1

            tile_condition = condition1 & condition2 & condition3 & condition4
            glathida_rgi_tile = glathida_rgi[tile_condition]

            tqdm.write(f"\t No. points found in tile {i+1}/{n_rgi_tiles}: {len(glathida_rgi_tile)}/{len(glathida_rgi)}")

            # If no point in tile go to next tile
            if len(glathida_rgi_tile)==0:
                continue

            ids_rgi = glathida_rgi_tile['GlaThiDa_ID'].unique().tolist()  # unique IDs in rgi tile

            trans_to_crs = Transformer.from_crs("EPSG:4326", tile_vx.rio.crs)
            ris_metre_millan = tile_vx.rio.resolution()[0]
            eps_millan = 10 * ris_metre_millan

            # loop over the unique IDs
            for id_rgi in tqdm(ids_rgi, total=len(ids_rgi), desc=f"rgi {rgi} tile {i+1}/{n_rgi_tiles} Glathida ID", leave=True):
                glathida_rgi_tile_id = glathida_rgi_tile.loc[glathida_rgi_tile['GlaThiDa_ID'] == id_rgi]
                indexes_rgi_tile_id = glathida_rgi_tile_id.index.tolist()

                lats = np.array(glathida_rgi_tile_id['POINT_LAT'])
                lons = np.array(glathida_rgi_tile_id['POINT_LON'])

                lons_crs, lats_crs = trans_to_crs.transform(lats, lons)
                swlat = np.amin(lats_crs)
                swlon = np.amin(lons_crs)
                nelat = np.amax(lats_crs)
                nelon = np.amax(lons_crs)
                deltalat = np.abs(swlat - nelat)
                deltalon = np.abs(swlon - nelon)

                # clip millan mosaic
                try:
                    focus_vx0 = tile_vx.rio.clip_box(minx=swlon-deltalon-eps_millan, miny=swlat-deltalat-eps_millan,
                                                          maxx=nelon+deltalon+eps_millan, maxy=nelat+deltalat+eps_millan)

                    focus_vy0 = tile_vy.rio.clip_box(minx=swlon-deltalon-eps_millan, miny=swlat-deltalat-eps_millan,
                                                          maxx=nelon+deltalon+eps_millan, maxy=nelat+deltalat+eps_millan)

                    focus_ith = tile_ith.rio.clip_box(minx=swlon-deltalon-eps_millan, miny=swlat-deltalat-eps_millan,
                                                          maxx=nelon+deltalon+eps_millan, maxy=nelat+deltalat+eps_millan)
                except:
                    tqdm.write(f'No millan data for rgi {rgi} GlaThiDa_ID {id_rgi} ')
                    # print(swlon - deltalon - eps, nelon + deltalon + eps, swlat - deltalat - eps, nelat + deltalat + eps)
                    continue

                # Interpolate vx, vy to remove nans (as much as possible). On the other hand we keep the nans
                # in the ice thickness ith.
                focus_vx = focus_vx0.rio.interpolate_na(method='linear').squeeze()
                focus_vy = focus_vy0.rio.interpolate_na(method='linear').squeeze()
                focus_ith = focus_ith.squeeze()

                plot_Millan_interpolated_tiles = False
                if plot_Millan_interpolated_tiles:
                    fig, axes = plt.subplots(1, 3)
                    ax1, ax2, ax3 = axes.flatten()
                    im1 = focus_vx0.plot(ax=ax1, cmap='viridis')
                    im2 = focus_vx.plot(ax=ax2, cmap='viridis')
                    s2 = ax2.scatter(x=lons_crs, y=lats_crs, s=20, c='k', alpha=.1, zorder=1)
                    im3 = focus_ith.plot(ax=ax3, cmap='viridis')
                    plt.show()

                # Calculate sigma in meters for adaptive gaussian fiter
                sigma_af_min, sigma_af_max = 100.0, 2000.0
                try:
                    # Each id_rgi may come with multiple area values and also nans (probably if all points outside glacier geometries)
                    area_id = glathida_rgi_tile_id['Area'].min()  # km2
                    lmax_id = glathida_rgi_tile_id['Lmax'].max()  # m
                    a = 1e6 * area_id / (np.pi * 0.5 * lmax_id)
                    sigma_af = int(min(max(a, sigma_af_min), sigma_af_max))
                    # print(area_id, lmax_id, a, value)
                except Exception as e:
                    sigma_af = sigma_af_min
                # Ensure that our value correctly in range [50.0, 2000.0]
                assert sigma_af_min <= sigma_af <= sigma_af_max, f"Value {sigma_af} is not within the range [{sigma_af_min}, {sigma_af_max}]"
                # print(f"Adaptive gaussian filter with sigma = {value} meters.")
                #print(sigma_af)



                # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
                num_px_sigma_50 = max(1, round(50 / ris_metre_millan))  # 1
                num_px_sigma_100 = max(1, round(100 / ris_metre_millan))  # 2
                num_px_sigma_150 = max(1, round(150 / ris_metre_millan))  # 3
                num_px_sigma_300 = max(1, round(300 / ris_metre_millan))  # 6
                num_px_sigma_450 = max(1, round(450 / ris_metre_millan))  # 6
                num_px_sigma_af = max(1, round(sigma_af / ris_metre_millan))

                # Apply filter to velocities
                focus_filter_vx_50 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_50, trunc=4.0)
                focus_filter_vx_100 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_100, trunc=4.0)
                focus_filter_vx_150 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_150, trunc=4.0)
                focus_filter_vx_300 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_300, trunc=4.0)
                focus_filter_vx_450 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_450, trunc=4.0)
                focus_filter_vx_af = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_af, trunc=3.0)

                focus_filter_vy_50 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_50, trunc=4.0)
                focus_filter_vy_100 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_100, trunc=4.0)
                focus_filter_vy_150 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_150, trunc=4.0)
                focus_filter_vy_300 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_300, trunc=4.0)
                focus_filter_vy_450 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_450, trunc=4.0)
                focus_filter_vy_af = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_af, trunc=3.0)


                # Mask back the filtered arrays
                focus_filter_vx_50 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_50)
                focus_filter_vx_100 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_100)
                focus_filter_vx_150 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_150)
                focus_filter_vx_300 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_300)
                focus_filter_vx_450 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_450)
                focus_filter_vx_af = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_af)
                focus_filter_vy_50 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_50)
                focus_filter_vy_100 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_100)
                focus_filter_vy_150 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_150)
                focus_filter_vy_300 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_300)
                focus_filter_vy_450 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_450)
                focus_filter_vy_af = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_af)

                # create xarrays of filtered velocities
                focus_filter_vx_50_ar = focus_vx.copy(deep=True, data=focus_filter_vx_50)
                focus_filter_vx_100_ar = focus_vx.copy(deep=True, data=focus_filter_vx_100)
                focus_filter_vx_150_ar = focus_vx.copy(deep=True, data=focus_filter_vx_150)
                focus_filter_vx_300_ar = focus_vx.copy(deep=True, data=focus_filter_vx_300)
                focus_filter_vx_450_ar = focus_vx.copy(deep=True, data=focus_filter_vx_450)
                focus_filter_vx_af_ar = focus_vx.copy(deep=True, data=focus_filter_vx_af)
                focus_filter_vy_50_ar = focus_vy.copy(deep=True, data=focus_filter_vy_50)
                focus_filter_vy_100_ar = focus_vy.copy(deep=True, data=focus_filter_vy_100)
                focus_filter_vy_150_ar = focus_vy.copy(deep=True, data=focus_filter_vy_150)
                focus_filter_vy_300_ar = focus_vy.copy(deep=True, data=focus_filter_vy_300)
                focus_filter_vy_450_ar = focus_vy.copy(deep=True, data=focus_filter_vy_450)
                focus_filter_vy_af_ar = focus_vy.copy(deep=True, data=focus_filter_vy_af)

                # Calculate the velocity gradients
                dvx_dx_ar, dvx_dy_ar = focus_filter_vx_300_ar.differentiate(coord='x'), focus_filter_vx_300_ar.differentiate(coord='y')
                dvy_dx_ar, dvy_dy_ar = focus_filter_vy_300_ar.differentiate(coord='x'), focus_filter_vy_300_ar.differentiate(coord='y')

                # Interpolate (note: nans can be produced near boundaries). This should be removed at the end.
                lons_crs = xarray.DataArray(lons_crs)
                lats_crs = xarray.DataArray(lats_crs)

                ith_data = focus_ith.interp(y=lats_crs, x=lons_crs, method="nearest").data
                vx_data = focus_vx.interp(y=lats_crs, x=lons_crs, method="nearest").data
                vy_data = focus_vy.interp(y=lats_crs, x=lons_crs, method="nearest").data
                vx_filter_50_data = focus_filter_vx_50_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vx_filter_100_data = focus_filter_vx_100_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vx_filter_150_data = focus_filter_vx_150_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vx_filter_300_data = focus_filter_vx_300_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vx_filter_450_data = focus_filter_vx_450_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vx_filter_af_data = focus_filter_vx_af_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vy_filter_50_data = focus_filter_vy_50_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vy_filter_100_data = focus_filter_vy_100_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vy_filter_150_data = focus_filter_vy_150_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vy_filter_300_data = focus_filter_vy_300_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vy_filter_450_data = focus_filter_vy_450_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                vy_filter_af_data = focus_filter_vy_af_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data

                dvx_dx_data = dvx_dx_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                dvx_dy_data = dvx_dy_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                dvy_dx_data = dvy_dx_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
                dvy_dy_data = dvy_dy_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data

                # some checks
                assert ith_data.shape == vx_data.shape == vy_data.shape, "Millan interp something wrong!"
                assert vx_filter_150_data.shape == vy_filter_150_data.shape, "Millan interp something wrong!"
                assert vx_filter_150_data.shape == vy_filter_300_data.shape, "Millan interp something wrong!"
                assert vx_filter_300_data.shape == vy_filter_150_data.shape, "Millan interp something wrong!"
                assert vx_filter_450_data.shape == vy_filter_450_data.shape, "Millan interp something wrong!"
                assert vx_filter_af_data.shape == vy_filter_af_data.shape, "Millan interp something wrong!"
                assert dvx_dx_data.shape == dvx_dy_data.shape, "Millan interp something wrong!"
                assert dvy_dx_data.shape == dvy_dy_data.shape, "Millan interp something wrong!"

                # Fill dataframe
                # add ith_m, vx, vy data to dataframe
                glathida.loc[indexes_rgi_tile_id, 'ith_m'] = ith_data
                glathida.loc[indexes_rgi_tile_id, 'vx'] = vx_data
                glathida.loc[indexes_rgi_tile_id, 'vy'] = vy_data
                glathida.loc[indexes_rgi_tile_id, 'vx_gf50'] = vx_filter_50_data
                glathida.loc[indexes_rgi_tile_id, 'vx_gf100'] = vx_filter_100_data
                glathida.loc[indexes_rgi_tile_id, 'vx_gf150'] = vx_filter_150_data
                glathida.loc[indexes_rgi_tile_id, 'vx_gf300'] = vx_filter_300_data
                glathida.loc[indexes_rgi_tile_id, 'vx_gf450'] = vx_filter_450_data
                glathida.loc[indexes_rgi_tile_id, 'vx_gfa'] = vx_filter_af_data
                glathida.loc[indexes_rgi_tile_id, 'vy_gf50'] = vy_filter_50_data
                glathida.loc[indexes_rgi_tile_id, 'vy_gf100'] = vy_filter_100_data
                glathida.loc[indexes_rgi_tile_id, 'vy_gf150'] = vy_filter_150_data
                glathida.loc[indexes_rgi_tile_id, 'vy_gf300'] = vy_filter_300_data
                glathida.loc[indexes_rgi_tile_id, 'vy_gf450'] = vy_filter_450_data
                glathida.loc[indexes_rgi_tile_id, 'vy_gfa'] = vy_filter_af_data
                glathida.loc[indexes_rgi_tile_id, 'dvx_dx'] = dvx_dx_data
                glathida.loc[indexes_rgi_tile_id, 'dvx_dy'] = dvx_dy_data
                glathida.loc[indexes_rgi_tile_id, 'dvy_dx'] = dvy_dx_data
                glathida.loc[indexes_rgi_tile_id, 'dvy_dy'] = dvy_dy_data

                # Plot
                ifplot_millan = False
                if ifplot_millan and np.any(np.isnan(vx_data)): ifplot_millan = True
                else: ifplot_millan = False
                if ifplot_millan:
                    lons_crs, lats_crs = lons_crs.to_numpy(), lats_crs.to_numpy()
                    fig, axes = plt.subplots(3, 3, figsize=(10, 5))
                    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes.flatten()

                    im1 = focus_vx.plot(ax=ax1, cmap='viridis', vmin=np.nanmin(focus_vx), vmax=np.nanmax(focus_vx))

                    s1 = ax1.scatter(x=lons_crs, y=lats_crs, s=15, c=vx_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=np.nanmin(focus_vx), vmax=np.nanmax(focus_vx), zorder=1)

                    s1_1 = ax1.scatter(x=lons_crs[np.argwhere(np.isnan(vx_data))], y=lats_crs[np.argwhere(np.isnan(vx_data))],
                                       s=15, c='magenta', zorder=1)

                    im2 = focus_vy.plot(ax=ax2, cmap='viridis', vmin=np.nanmin(focus_vy), vmax=np.nanmax(focus_vy))
                    s2 = ax2.scatter(x=lons_crs, y=lats_crs, s=15, c=vy_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=np.nanmin(focus_vy), vmax=np.nanmax(focus_vy), zorder=1)
                    s2_1 = ax2.scatter(x=lons_crs[np.argwhere(np.isnan(vy_data))], y=lats_crs[np.argwhere(np.isnan(vy_data))],
                                       s=15, c='magenta', zorder=1)

                    im3 = focus_ith.plot(ax=ax3, cmap='viridis', vmin=0, vmax=np.nanmax(ith_data))
                    s3 = ax3.scatter(x=lons_crs, y=lats_crs, s=15, c=ith_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=0, vmax=np.nanmax(ith_data), zorder=1)
                    s3_1 = ax3.scatter(x=lons_crs[np.argwhere(np.isnan(ith_data))], y=lats_crs[np.argwhere(np.isnan(ith_data))],
                                       s=15, c='magenta', zorder=1)

                    im4 = focus_filter_vx_150_ar.plot(ax=ax4, cmap='viridis', vmin=np.nanmin(focus_vx), vmax=np.nanmax(focus_vx))
                    s4 = ax4.scatter(x=lons_crs, y=lats_crs, s=15, c=vx_filter_150_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=np.nanmin(focus_vx), vmax=np.nanmax(focus_vx), zorder=1)
                    s4_1 = ax4.scatter(x=lons_crs[np.argwhere(np.isnan(vx_filter_150_data))],
                                       y=lats_crs[np.argwhere(np.isnan(vx_filter_150_data))], s=15, c='magenta', zorder=1)

                    im5 = focus_filter_vy_150_ar.plot(ax=ax5, cmap='viridis', vmin=np.nanmin(focus_vy),
                                                      vmax=np.nanmax(focus_vy))
                    s5 = ax5.scatter(x=lons_crs, y=lats_crs, s=15, c=vy_filter_150_data, ec=(0, 0, 0, 0.3), cmap='viridis',
                                     vmin=np.nanmin(focus_vy), vmax=np.nanmax(focus_vy), zorder=1)
                    s5_1 = ax5.scatter(x=lons_crs[np.argwhere(np.isnan(vy_filter_150_data))],
                                       y=lats_crs[np.argwhere(np.isnan(vy_filter_150_data))], s=15,
                                       c='magenta', zorder=1)

                    ax6.axis("off")
                    ax9.axis("off")

                    im7 = dvx_dx_ar.plot(ax=ax7, cmap='viridis', )
                    im8 = dvy_dy_ar.plot(ax=ax8, cmap='viridis', )

                    ax1.title.set_text('vx')
                    ax2.title.set_text('vy')
                    ax3.title.set_text('ice thickness')
                    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
                        ax.set(xlabel='', ylabel='')
                    plt.tight_layout()
                    plt.show()


        # How many nans we have produced from the interpolation
        glathida_rgi_ = glathida.loc[glathida['RGI'] == rgi]
        tqdm.write(
            f"\t From rgi {rgi} the no. nans in vx/vy/ith/vx300/vy300/etc: {np.sum(np.isnan(glathida_rgi_['vx']))}/{np.sum(np.isnan(glathida_rgi_['vy']))}"
            f"/{np.sum(np.isnan(glathida_rgi_['ith_m']))}/{np.sum(np.isnan(glathida_rgi_['vx_gf300']))}/{np.sum(np.isnan(glathida_rgi_['vy_gf300']))}/"
            f"{np.sum(np.isnan(glathida_rgi_['dvx_dx']))}/{np.sum(np.isnan(glathida_rgi_['dvx_dy']))}/{np.sum(np.isnan(glathida_rgi_['dvy_dx']))}/"
            f"{np.sum(np.isnan(glathida_rgi_['dvy_dy']))}")


    print(f"Millan done in {(time.time()-tm)/60} min.")
    return glathida

"""Add distance from border using Millan's velocity"""
def add_dist_from_border_in_out(glathida, path_millan_velocity):
    print("Adding distance to border...")

    if ('dist_from_border_km' in list(glathida) or 'outsider' in list(glathida)):
        print('Variable already in dataframe.')
        return glathida

    glathida['dist_from_border_km'] = [np.nan]*len(glathida)
    glathida['outsider'] = [np.nan]*len(glathida)
    col_index_dist = glathida.columns.get_loc('dist_from_border_km')
    col_index_outsider = glathida.columns.get_loc('outsider')

    regions=[3, 7, 8, 11, 18]

    for rgi in tqdm(regions, total=len(regions), leave=True):
        tqdm.write(f'rgi: {rgi}')

        files_v = glob(path_millan_velocity + 'RGI-{}/V_RGI-{}*'.format(rgi, rgi))
        # merge v files to create vy mosaic
        xds_4326 = []
        for tiffile in files_v:
            xds = rioxarray.open_rasterio(tiffile, masked=False)
            xds.rio.write_nodata(np.nan, inplace=True)
            xds_4326.append(xds.rio.reproject("EPSG:4326")) #todo: need to change reproject resampling method
        mosaic_v = merge.merge_arrays(xds_4326)
        #mosaic_v.plot()
        #plt.show()

        ris_ang = mosaic_v.rio.resolution()[0]
        eps = 40 * ris_ang

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi] # collapse glathida to specific rgi
        indexes_all = glathida_rgi.index.tolist()

        lats = np.array(glathida_rgi['POINT_LAT'])
        lons = np.array(glathida_rgi['POINT_LON'])

        # loop over indexes of specific rgi
        for n, index in tqdm(enumerate(indexes_all), total=len(lats), leave=True):

            # get lat lon of measurement
            lon = lons[n]
            lat = lats[n]

            # clip millan mosaic
            count_resize = 1
            deltalat = eps
            deltalon = eps
            try:
                focus = mosaic_v.rio.clip_box(minx=lon - deltalon - eps, miny=lat - deltalat - eps,
                                              maxx=lon + deltalon + eps, maxy=lat + deltalat + eps)
            except:
                tqdm.write(f'non ci sono dati di millan per rgi: {rgi} index: {index} lat: {lat} lon: {lon}')
                continue

            dataxarray = focus.data.squeeze()
            nanpos = np.argwhere(np.isnan(dataxarray)).T

            # if no nan are present clip a bigger region until we find nans
            while not nanpos.size:
                count_resize += 1
                deltalat = 2 * deltalat
                deltalon = 2 * deltalon
                focus = mosaic_v.rio.clip_box(minx=lon - deltalon - eps, miny=lat - deltalat - eps,
                                              maxx=lon + deltalon + eps, maxy=lat + deltalat + eps)
                dataxarray = focus.data.squeeze()
                nanpos = np.argwhere(np.isnan(dataxarray)).T

            coordx = focus.coords['x'].to_numpy()[nanpos[1]]
            coordy = focus.coords['y'].to_numpy()[nanpos[0]]
            A = np.array([coordx, coordy]).T
            ris_ang = np.abs(focus.rio.resolution()[0])

            kdtree = spatial.KDTree(A).query(np.array([lon, lat]).T)

            if kdtree[0] > ris_ang / np.sqrt(2):
                nnpt = A[kdtree[1]]
                dist = haversine(lon, lat, nnpt[0], nnpt[1])
                outsider = 0
            else: # se il punto trovato e' di fatto fuori al campo di velocita'
                dist = np.nan
                outsider = 1

            #tqdm.write(f'rgi:{rgi} n:{n} index:{index} A size: {A.shape}focus size:{focus.rio.width}x{focus.rio.height} '
            #           f'risized times: {count_resize} dist:{dist:.3f} km')

            # add dist and outsider to dataframe
            glathida.iat[index, col_index_dist] = dist
            glathida.iat[index, col_index_outsider] = outsider
            # check if the calculated distance has been added to the dataframe correctly
            #print(glathida['POINT_LAT'].iloc[index], glathida['POINT_LAT'].iloc[index], glathida['dist_from_border_km'].iloc[index])

            # plot randomly
            ifplot = False
            #p = random.randrange(0, 10000, 1)
            #if p == 50: ifplot = True
            #if dist>3: ifplot = True
            if ifplot:
                fig, ax1 = plt.subplots()
                im1 = focus.plot(cmap='Blues', vmin=focus.min(), vmax=focus.max())
                if outsider == 0:
                    s_measurement = ax1.scatter(x=lon, y=lat, s=30, c='magenta')
                    s_closest     = ax1.scatter(x=nnpt[0], y=nnpt[1], s=30, c='lime')
                elif outsider == 1:
                    s_measurement = ax1.scatter(x=lon, y=lat, s=30, c='k')
                plt.show()

    return glathida

"""Add distance from border using glacier geometries"""
def add_dist_from_boder_using_geometries(glathida):
    print("Adding distance to border using a geometrical approach...")
    # Note that if the point is inside a nunatak the distance will be set to nan.

    if ('dist_from_border_km_geom' in list(glathida)):
        print('Variable already in dataframe.')
        return glathida

    glathida['dist_from_border_km_geom'] = [np.nan] * len(glathida)

    def add_new_neighbors(neigbords, df):
        """ I give a list of neighbors and I should return a new list with added neighbors"""
        for id in neigbords:
            neighbors_wrt_id = df[df['RGIId_1'] == id]['RGIId_2'].unique()
            neigbords = np.append(neigbords, neighbors_wrt_id)
        neigbords = np.unique(neigbords)
        return neigbords

    def find_cluster_RGIIds(id, df):
        neighbors0 = np.array([id])
        len0 = len(neighbors0)
        neighbors1 = add_new_neighbors(neighbors0, df)
        len1 = len(neighbors1)
        while (len1 > len0):
            len0 = len1
            neighbors1 = add_new_neighbors(neighbors1, df)
            len1 = len(neighbors1)
        if (len(neighbors1)) ==1: return None
        else: return neighbors1

    regions = [1,3,4,7,8,11,18]

    # loop over regions
    for rgi in tqdm(regions, total=len(regions), desc='RGI',  leave=True):
        t_ = time.time()

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]
        tqdm.write(f"Region RGI: {rgi}, {len(glathida_rgi['RGIId'].dropna())} points")

        rgi_ids = glathida_rgi['RGIId'].dropna().unique().tolist()
        # print(f"We have {len(rgi_ids)} valid glaciers and {len(glathida_rgi)} rows "
        #      f"of which {glathida_rgi['RGIId'].isna().sum()} points without a glacier id (hence nan)"
        #      f"and {glathida_rgi['RGIId'].notna().sum()} points with valid glacier id")

        oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')  # get rgi region shp
        oggm_rgi_intersects_shp = utils.get_rgi_intersects_region_file(f"{rgi:02d}", version='62')

        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)  # get rgi dataset of glaciers
        oggm_rgi_intersects = gpd.read_file(oggm_rgi_intersects_shp)  # get rgi dataset of glaciers intersects

        # loop over glaciers
        # Note: It is important to note that since rgi_ids do not contain nans, looping over it automatically
        # selects only the points inside glaciers (and not those outside)
        for rgi_id in tqdm(rgi_ids, total=len(rgi_ids), desc=f"Glaciers in rgi {rgi}", leave=True):

            multipolygon = False

            # Get glacier glathida and oggm datasets
            try:
                glathida_id = glathida_rgi.loc[glathida_rgi['RGIId'] == rgi_id] # glathida dataset

                gl_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == rgi_id] # oggm dataset
                gl_geom = gl_df['geometry'].item()  # glacier geometry Polygon
                gl_geom_ext = Polygon(gl_geom.exterior)  # glacier geometry Polygon
                gl_geom_nunataks_list = [Polygon(nunatak) for nunatak in gl_geom.interiors]  # list of nunataks Polygons
                #print(f"Glacier {rgi_id} found and its {len(glathida_id)} points contained.")
                assert len(gl_df) == 1, "Check this please."
                # Get the UTM EPSG code from glacier center coordinates
                cenLon, cenLat = gl_df['CenLon'].item(), gl_df['CenLat'].item()
                _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(cenLat, cenLon)

            except Exception as e:
                print(f"Error {e} with glacier {rgi_id}. It was not found so it be skipped.")
                continue

            # intersects of glacier (need only for plotting purposes)
            gl_intersects = oggm.utils.get_rgi_intersects_entities([rgi_id], version='62')

            # Calculate intersects of all glaciers in the cluster
            list_cluster_RGIIds = find_cluster_RGIIds(rgi_id, oggm_rgi_intersects)
            #print(f"List of glacier cluster: {list_cluster_RGIIds}")

            if list_cluster_RGIIds is not None:
                # (need only for plotting purposes)
                cluster_intersects = oggm.utils.get_rgi_intersects_entities(list_cluster_RGIIds, version='62')
            else: cluster_intersects = None

            # Now calculate the geometries
            if list_cluster_RGIIds is None:  # Case 1: isolated glacier
                #print(f"Isolated glacier")
                exterior_ring = gl_geom.exterior  # shapely.geometry.polygon.LinearRing
                interior_rings = gl_geom.interiors  # shapely.geometry.polygon.InteriorRingSequence of polygon.LinearRing
                geoseries_geometries_4326 = gpd.GeoSeries([exterior_ring] + list(interior_rings), crs="EPSG:4326")
                geoseries_geometries_epsg = geoseries_geometries_4326.to_crs(epsg=glacier_epsg)

            elif list_cluster_RGIIds is not None:  # Case 2: cluster of glaciers with ice divides
                #print(f"Cluster of glaciers with ice divides.")
                cluster_geometry_list = []
                for gl_neighbor_id in list_cluster_RGIIds:
                    gl_neighbor_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_neighbor_id]
                    gl_neighbor_geom = gl_neighbor_df['geometry'].item()
                    cluster_geometry_list.append(gl_neighbor_geom)

                # Combine into a series of all glaciers in the cluster
                cluster_geometry_4326 = gpd.GeoSeries(cluster_geometry_list, crs="EPSG:4326")

                # Now remove all ice divides
                cluster_geometry_no_divides_4326 = gpd.GeoSeries(cluster_geometry_4326.unary_union, crs="EPSG:4326")
                cluster_geometry_no_divides_epsg = cluster_geometry_no_divides_4326.to_crs(epsg=glacier_epsg)
                if cluster_geometry_no_divides_epsg.item().geom_type == 'Polygon':
                    cluster_exterior_ring = [cluster_geometry_no_divides_epsg.item().exterior]  # shapely.geometry.polygon.LinearRing
                    cluster_interior_rings = list(cluster_geometry_no_divides_epsg.item().interiors) # shapely.geometry.polygon.LinearRing
                elif cluster_geometry_no_divides_epsg.item().geom_type == 'MultiPolygon':
                    polygons = list(cluster_geometry_no_divides_epsg.item().geoms)
                    cluster_exterior_ring = [polygon.exterior for polygon in polygons] # list of shapely.geometry.polygon.LinearRing
                    num_multipoly = len(cluster_exterior_ring)
                    cluster_interior_ringSequences = [polygon.interiors for polygon in polygons] # list of shapely.geometry.polygon.InteriorRingSequence
                    cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence] # list of shapely.geometry.polygon.LinearRing
                    multipolygon = True
                else: raise ValueError("Unexpected geometry type. Please check.")

                # Create a geoseries of all external and internal geometries (NB I set here the glacier_epsg)
                geoseries_geometries_epsg = gpd.GeoSeries(cluster_exterior_ring + cluster_interior_rings).set_crs(epsg=glacier_epsg)

            # Get all points and create Geopandas geoseries and convert to glacier center UTM
            # Note: a delicate issue is that technically each point may have its own UTM zone.
            # To keep things consistent I decide to project all of them to the glacier center UTM.
            lats = glathida_id['POINT_LAT'].tolist()
            lons = glathida_id['POINT_LON'].tolist()
            list_points = [Point(lon, lat) for (lon, lat) in zip(lons, lats)]
            geoseries_points_4326 = gpd.GeoSeries(list_points, crs="EPSG:4326")
            geoseries_points_epsg = geoseries_points_4326.to_crs(epsg=glacier_epsg)

            # List of distances for glacier_id
            glacier_id_dist = []

            # Decide which method to use (default should be method_KDTree_spatial_index)
            method_geopandas_spatial_index = False
            method_KDTree_spatial_index = True
            method_geopandas_distances = False

            if method_geopandas_spatial_index:
                # Create spatial index for the geometries
                sindex_id = geoseries_geometries_epsg.sindex

            if method_KDTree_spatial_index:
                # 1. Extract all coordinates from the GeoSeries geometries for the current glacier
                geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in geoseries_geometries_epsg.geometry])
                # 2. instantiate kdtree
                kdtree = KDTree(geoms_coords_array)

            for i, (idx, lon, lat) in tqdm(enumerate(zip(glathida_id.index, lons, lats)), total=len(lons), desc='Points', leave=False):

                # Make check 0.
                make_check0 = False
                if make_check0 and i==0:
                    lat_check = glathida_id.loc[idx, 'POINT_LAT']
                    lon_check = glathida_id.loc[idx, 'POINT_LON']
                    #print(lon_check, lat_check, lon, lat)

                # Make check 1.
                make_check1 = True
                if make_check1:
                    is_inside = gl_geom_ext.contains(Point(lon, lat))
                    assert is_inside is True, f"The point is expected to be inside but is outside glacier."

                make_check2 = False
                # Note that the UTM zone of one point may not be the UTM zone of the glacier center
                # I think projecting the point using the UTM zone of the glacier center is fine.
                if make_check2:
                    easting, nothing, zonenum, zonelett, epsg = from_lat_lon_to_utm_and_epsg(lat, lon)
                    if epsg != glacier_epsg:
                        # todo: maybe need to correct for this.
                        print(f"Note differet UTM zones. Point espg {epsg} and glacier center epsg {glacier_epsg}.")

                # Decide whether point is inside a nunatak. If yes set the distance to nan
                is_nunatak = any(nunatak.contains(Point(lon, lat)) for nunatak in gl_geom_nunataks_list)
                if is_nunatak is True:
                    min_dist = np.nan

                # if not
                else:
                    # get shapely Point
                    point_epsg = geoseries_points_epsg.iloc[i]

                    # Method 1 with geopandas spatial index (fast)
                    if method_geopandas_spatial_index:

                        # Find the index of the nearest geometry
                        nearest_idx = sindex_id.nearest(point_epsg.bounds)
                        # Get the nearest geometry (NB may consists of more than one geometry)
                        nearest_geometries = geoseries_geometries_epsg.iloc[nearest_idx]
                        # Calculate the distance between the closest geometry and the point
                        min_distances = nearest_geometries.distance(point_epsg)
                        # Find the index of the row with the minimum distance
                        min_idx = min_distances.idxmin()

                        # Get the minimum distance and corresponding geometry
                        min_dist_spatial_index = min_distances.loc[min_idx]
                        nearest_geometry = nearest_geometries.loc[min_idx]
                        #print(min_distances)
                        #print(min_dist_spatial_index)
                        #print(nearest_geometries)

                        # Find the nearest point on the boundary of the polygon
                        get_closest_point = True
                        if get_closest_point:
                            nearest_point_on_boundary, nearest_point_point = nearest_points(nearest_geometry, point_epsg)
                            # Calculate the minimum distance again, just to verify
                            #min_distance_check = nearest_point_on_boundary.distance(point_epsg)
                            #print(min_distance_check)

                    # Method 2 with KDTree
                    if method_KDTree_spatial_index:

                        point_array = np.array(point_epsg.coords)  # (1,2)

                        # Perform nearest neighbor search for each point and calculate minimum distances
                        # both distances and indices have shape (1, len(geoseries_geometries_epsg))
                        distances, indices = kdtree.query(point_array, k=len(geoseries_geometries_epsg))
                        min_dist_KDTree = distances[0,0] # np.min(distances, axis=1)) or equivalently distances[:,0]
                        closest_point_index = indices[0, 0]
                        closest_point = Point(geoms_coords_array[closest_point_index])
                        # todo: convert to lat lon and return it add slope calculation method
                        #tqdm.write(f"{min_dist_KDTree}")
                        #tqdm.write(f"{closest_point, nearest_point_on_boundary}")

                    # Method 3 (exact but slow): Calculate the distances between such point and all glacier geometries.
                    if method_geopandas_distances:
                        min_distances_point_geometries = geoseries_geometries_epsg.distance(point_epsg)
                        min_dist_geopandas_distances = np.min(min_distances_point_geometries)  # unit UTM: m

                        # To debug we want to check what point corresponds to the minimum distance.
                        debug_distance = True
                        if debug_distance:
                            min_distance_index = min_distances_point_geometries.idxmin()
                            nearest_line = geoseries_geometries_epsg.loc[min_distance_index]
                            nearest_point_on_line = nearest_line.interpolate(nearest_line.project(point_epsg))


                # Fill distance list for glacier id of point
                glacier_id_dist.append(min_dist_KDTree/1000.)

                # For debugging
                if method_KDTree_spatial_index and method_geopandas_spatial_index and method_geopandas_distances:
                    if min_dist_KDTree>300 and abs(min_dist_KDTree-min_dist_geopandas_distances)/min_dist_geopandas_distances >.5:
                        plot_calculate_distance = True
                        print(min_dist_KDTree, min_dist_geopandas_distances, min_dist_spatial_index)
                    else: plot_calculate_distance = False
                else: plot_calculate_distance = False

                # Plot
                #r = random.uniform(0,1)
                #if (plot_calculate_distance and list_cluster_RGIIds is not None and r<1.0):
                if plot_calculate_distance:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    ax1.plot(*gl_geom_ext.exterior.xy, lw=1, c='magenta', zorder=4)
                    for interior in gl_geom.interiors:
                        ax1.plot(*interior.xy, lw=1, c='blue')

                    # Plot boundaries (only external periphery) of all glaciers in the cluster
                    if list_cluster_RGIIds is not None:
                        for gl_neighbor_id in list_cluster_RGIIds:
                            gl_neighbor_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == gl_neighbor_id]
                            gl_neighbor_geom = gl_neighbor_df['geometry'].item()  # glacier geometry Polygon
                            ax1.plot(*gl_neighbor_geom.exterior.xy, lw=1, c='orange', zorder=0)

                    # Plot intersections of central glacier with its neighbors
                    for k, intersect in enumerate(gl_intersects['geometry']):  # Linestring gl_intersects
                        ax1.plot(*intersect.xy, lw=1, color='k')

                    # Plot intersections of all glaciers in the cluster
                    if cluster_intersects is not None:
                        for k, intersect in enumerate(cluster_intersects['geometry']):
                            ax1.plot(*intersect.xy, lw=1, color='k')  # np.random.rand(3)

                        # Plot cluster ice divides removed
                        if multipolygon:
                            polygons = list(cluster_geometry_no_divides_4326.item().geoms)
                            cluster_exterior_ring = [polygon.exterior for polygon in polygons]  # list of shapely.geometry.polygon.LinearRing
                            cluster_interior_ringSequences = [polygon.interiors for polygon in polygons]  # list of shapely.geometry.polygon.InteriorRingSequence
                            cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence]  # list of shapely.geometry.polygon.LinearRing
                            for exterior in cluster_exterior_ring:
                                ax1.plot(*exterior.xy , lw=1, c='red', zorder=3)
                            for interior in cluster_interior_rings:
                                ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

                        else:
                            ax1.plot(*cluster_geometry_no_divides_4326.item().exterior.xy, lw=1, c='red', zorder=3)
                            for interior in cluster_geometry_no_divides_4326.item().interiors:
                                ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

                    if is_nunatak: ax1.scatter(lon, lat, s=50, lw=2, c='b')
                    else: ax1.scatter(lon, lat, s=50, lw=2, c='r', ec='r')

                    if multipolygon:
                        for i_poly in range(num_multipoly):
                            ax2.plot(*geoseries_geometries_epsg.loc[i_poly].xy, lw=1, c='red')  # first num_multipoly are outside borders
                        for inter in geoseries_geometries_epsg.loc[num_multipoly:]:  # all interiors if present
                            ax2.plot(*inter.xy, lw=1, c='blue')

                    else:
                        ax2.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='red')  # first entry is outside border
                        for inter in geoseries_geometries_epsg.loc[1:]:  # all interiors if present
                            ax2.plot(*inter.xy, lw=1, c='blue')

                    if is_nunatak: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='b', zorder=5)
                    else: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='r', ec='r', zorder=5)

                    if method_geopandas_distances and debug_distance:
                        ax2.scatter(*nearest_point_on_line.xy, s=50, lw=2, c='g', zorder=5)

                    if method_geopandas_spatial_index and get_closest_point:
                        ax2.scatter(x=nearest_point_on_boundary.x, y=nearest_point_on_boundary.y, s=40, lw=2, c='y', zorder=5)

                    if method_KDTree_spatial_index: ax2.scatter(x=closest_point.x, y=closest_point.y, s=30, lw=2, c='k', zorder=5)

                    ax1.set_title('EPSG 4326')
                    ax2.set_title(f'EPSG {glacier_epsg}')
                    plt.show()


            assert len(glacier_id_dist)==len(glathida_id.index), "Number of measurements does not match index length"
            #tqdm.write(f"Finished glacier {rgi_id} with {len(glacier_id_dist)} measurements")

            # Fill dataframe with distances from glacier rgi_id
            glathida.loc[glathida_id.index, 'dist_from_border_km_geom'] = glacier_id_dist

        tqdm.write(f"Finished region {rgi} in {(time.time()-t_)/60} mins.")

    return glathida

"""Add RGIId and other OGGM stats like glacier area"""
def add_RGIId_and_OGGM_stats(glathida, path_OGGM_folder):
    # Note: points that are outside any glaciers will have nan to RGIId (and the other features)

    print("Adding OGGM's stats method...")
    if (any(ele in list(glathida) for ele in ['RGIId', 'Area'])):
        print('Variables RGIId/Area etc already in dataframe.')
        return glathida

    glathida['RGIId'] = [np.nan] * len(glathida)
    glathida['Area'] = [np.nan] * len(glathida)
    glathida['Zmin'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Zmax'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Zmed'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Slope'] = [np.nan] * len(glathida)
    glathida['Lmax'] = [np.nan] * len(glathida) # -9 missing values, see https://essd.copernicus.org/articles/14/3889/2022/essd-14-3889-2022.pdf
    glathida['Form'] = [np.nan] * len(glathida) # 9 Not assigned
    glathida['TermType'] = [np.nan] * len(glathida) # 9 Not assigned
    glathida['Aspect'] = [np.nan] * len(glathida) # -9 bad values

    # Define this function for the parallelization
    def check_contains(point, geometry):
        return geometry.contains(point)

    regions = [1,3,4,7,8,11,18]

    for rgi in regions:
        # get OGGM's dataframe of rgi glaciers
        oggm_rgi_shp = glob(f'{path_OGGM_folder}/rgi/RGIV62/{rgi:02d}*/{rgi:02d}*.shp')[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
        #pd.set_option('display.max_columns', None)
        #print(oggm_rgi_glaciers.describe())
        #print(oggm_rgi_glaciers['Form'].value_counts())
        #print(oggm_rgi_glaciers['Aspect'].value_counts())
        #fig, ax = plt.subplots()
        #ax.hist(oggm_rgi_glaciers['Aspect'], bins=400)
        #plt.show()
        #print(oggm_rgi_glaciers['TermType'].value_counts())
        #print(oggm_rgi_glaciers['Surging'].value_counts())
        #input('wait')
        #print(oggm_rgi_glaciers.head(5))

        #print(f'rgi: {rgi}. Imported OGGMs {oggm_rgi_shp} dataframe of {len(oggm_rgi_glaciers)} glaciers')

        # Glathida
        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        # loop sui ghiacciai di oggm
        for i, ind in tqdm(enumerate(oggm_rgi_glaciers.index), total=len(oggm_rgi_glaciers), desc=f"glaciers in rgi {rgi}", leave=True, position=0):

            glacier_geometry = oggm_rgi_glaciers.at[ind, 'geometry']
            glacier_RGIId = oggm_rgi_glaciers.at[ind, 'RGIId']
            glacier_area = oggm_rgi_glaciers.at[ind, 'Area']
            glacier_zmin = oggm_rgi_glaciers.at[ind, 'Zmin']
            glacier_zmax = oggm_rgi_glaciers.at[ind, 'Zmax']
            glacier_zmed = oggm_rgi_glaciers.at[ind, 'Zmed']
            glacier_slope = oggm_rgi_glaciers.at[ind, 'Slope']
            glacier_lmax = oggm_rgi_glaciers.at[ind, 'Lmax']
            glacier_form = oggm_rgi_glaciers.at[ind, 'Form']
            glacier_termtype = oggm_rgi_glaciers.at[ind, 'TermType']
            glacier_aspect = oggm_rgi_glaciers.at[ind, 'Aspect']

            # calculate the area using pyproj and shapely for comparison with oggm
            # area_pyproj = abs(Geod(ellps="WGS84").geometry_area_perimeter(shapely.wkt.loads(str(glacier_geometry)))[0])*1.e-6 #km2

            llx, lly, urx, ury = glacier_geometry.bounds

            # dataframe of only points inside the raster bounds
            df_points_in_bound = glathida_rgi.loc[(glathida_rgi['POINT_LON'] >= llx) &
                                                  (glathida_rgi['POINT_LON'] <= urx) &
                                                  (glathida_rgi['POINT_LAT'] >= lly) &
                                                  (glathida_rgi['POINT_LAT'] <= ury)]

            # if no points inside the bound no reason to go further
            if (len(df_points_in_bound)==0): continue

            lats_in_bound = df_points_in_bound['POINT_LAT']
            lons_in_bound = df_points_in_bound['POINT_LON']
            points_in_bound = [Point(ilon, ilat) for (ilon, ilat) in zip(lons_in_bound, lats_in_bound)]
            #print(f'RGIId {glacier_RGIId} No. point in bound: {len(df_points_in_bound)}')

            # mask True/False to decide whether the points are inside the glacier geometry
            # NOTE: PARALLELIZATION HERE
            mask_points_in_glacier = Parallel(n_jobs=-1)(delayed(check_contains)(point, glacier_geometry) for point in points_in_bound)
            mask_points_in_glacier = np.array(mask_points_in_glacier)
            #mask_points_in_glacier_0 = glacier_geometry.contains(points_in_bound)
            #print(np.array_equal(mask_points_in_glacier_0, mask_points_in_glacier))

            # select only those points inside the glacier
            df_poins_in_glacier = df_points_in_bound[mask_points_in_glacier]
            # if no points inside the glacier move on
            if len(df_poins_in_glacier) == 0: continue

            #print(f'RGIId {glacier_RGIId} No. point in glacier: {len(df_poins_in_glacier)}')
            # at this point we should have found the points inside the glacier

            ifplot = False
            if ifplot:
                lats_in_glacier = df_poins_in_glacier['POINT_LAT']
                lons_in_glacier = df_poins_in_glacier['POINT_LON']
                fig, ax1 = plt.subplots()
                s1 = ax1.scatter(x=lons_in_bound, y=lats_in_bound, s=5, c='k')
                s2 = ax1.scatter(x=lons_in_glacier, y=lats_in_glacier, s=5, c='magenta', zorder=1)
                ax1.plot(*glacier_geometry.exterior.xy, c='magenta')
                plt.show()

            # some checks before returning values to dataframe
            assert glacier_zmin != -999, "Zmin should not be -999"
            assert glacier_zmax != -999, "Zmax should not be -999"
            assert glacier_zmed != -999, "Zmed should not be -999"
            assert glacier_lmax != -9, "Lmax should not be -9"
            assert glacier_form != 9, "Form should not be 9 (not assigned)"
            assert glacier_termtype != 9, "TermType should not be 9 (not assigned)"
            #assert glacier_aspect != -9, "Aspect should not be -9"

            # Data imputation (found needed for Greenland)
            if glacier_aspect == -9: glacier_aspect = 0

            assert not np.any(np.isnan(np.array([glacier_area, glacier_zmin, glacier_zmax,
                                                     glacier_zmed, glacier_slope, glacier_lmax,
                                                 glacier_form, glacier_termtype, glacier_aspect]))), \
                "Found nan in some variables in method add_RGIId_and_OGGM_stats. Check why this value appeared."

            # Add to dataframe
            glathida.loc[df_poins_in_glacier.index, 'RGIId'] = glacier_RGIId
            glathida.loc[df_poins_in_glacier.index, 'Area'] = glacier_area
            glathida.loc[df_poins_in_glacier.index, 'Zmin'] = glacier_zmin
            glathida.loc[df_poins_in_glacier.index, 'Zmax'] = glacier_zmax
            glathida.loc[df_poins_in_glacier.index, 'Zmed'] = glacier_zmed
            glathida.loc[df_poins_in_glacier.index, 'Slope'] = glacier_slope
            glathida.loc[df_poins_in_glacier.index, 'Lmax'] = glacier_lmax
            glathida.loc[df_poins_in_glacier.index, 'Form'] = glacier_form
            glathida.loc[df_poins_in_glacier.index, 'TermType'] = glacier_termtype
            glathida.loc[df_poins_in_glacier.index, 'Aspect'] = glacier_aspect

    return glathida

"""Add Farinotti's ith"""
def add_farinotti_ith(glathida, path_farinotti_icethickness):
    print("Adding Farinotti ice thickness...")

    if ('ith_f' in list(glathida)):
        print('Variable already in dataframe.')
        return glathida

    glathida['ith_f'] = [np.nan] * len(glathida)

    regions = [1,3,4,7,8,11,18]

    for rgi in tqdm(regions, total=len(regions), leave=True):

        # get dataframe of rgi glaciers from oggm
        oggm_rgi_shp = glob(f'/home/nico/OGGM/rgi/RGIV62/{rgi:02d}*/{rgi:02d}*.shp')[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
        oggm_rgi_glaciers_geoms = oggm_rgi_glaciers['geometry']
        print(f'rgi: {rgi}. Imported OGGMs {oggm_rgi_shp} dataframe of {len(oggm_rgi_glaciers_geoms)} glaciers')

        # Glathida
        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]
        indexes_all = glathida_rgi.index.tolist()
        lats = np.array(glathida_rgi['POINT_LAT'])
        lons = np.array(glathida_rgi['POINT_LON'])
        lats_xar = xarray.DataArray(lats)
        lons_xar = xarray.DataArray(lons)

        tqdm.write(f'rgi: {rgi}. Glathida: {len(lats)} points')

        # Import farinotti ice thickness files
        files_names_farinotti = sorted(glob(path_farinotti_icethickness + f'composite_thickness_RGI60-{rgi:02d}/RGI60-{rgi:02d}/*'))
        list_glaciers_farinotti_4326 = []

        for n, tiffile in tqdm(enumerate(files_names_farinotti), total=len(files_names_farinotti), leave=True):

            glacier_name = tiffile.split('/')[-1].replace('_thickness.tif', '')
            try:
                # Farinotti has solutions for glaciers that no longer exist in rgi/oggm (especially some in rgi 4)
                # See page 28 of https://www.glims.org/RGI/00_rgi60_TechnicalNote.pdf
                glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name, 'geometry'].item()
            except ValueError:
                tqdm.write(f"{glacier_name} not present in OGGM's RGI v6.")
                continue

            file_glacier_farinotti = rioxarray.open_rasterio(tiffile, masked=False)
            file_glacier_farinotti = file_glacier_farinotti.where(file_glacier_farinotti != 0.0) # set to nan outside glacier
            file_glacier_farinotti.rio.write_nodata(np.nan, inplace=True)

            #todo: need to likely change reproject resampling method
            file_glacier_farinotti_4326 = file_glacier_farinotti.rio.reproject("EPSG:4326")
            file_glacier_farinotti_4326.rio.write_nodata(np.nan, inplace=True)
            bounds_4326 = file_glacier_farinotti_4326.rio.bounds()

            # dataframe of only points inside the raster bounds
            df_points_in_bound = glathida_rgi.loc[(glathida_rgi['POINT_LON']>=bounds_4326[0]) &
                                        (glathida_rgi['POINT_LON']<=bounds_4326[2]) &
                                        (glathida_rgi['POINT_LAT']>=bounds_4326[1]) &
                                        (glathida_rgi['POINT_LAT']<=bounds_4326[3])]

            # if no points inside the bound no reason to go further
            if len(df_points_in_bound) == 0: continue

            # we want to mosaic only those raster that do contain glathida data inside the bound
            list_glaciers_farinotti_4326.append(file_glacier_farinotti_4326)

            lats_in_bound = df_points_in_bound['POINT_LAT']
            lons_in_bound = df_points_in_bound['POINT_LON']
            points_in_bound = [Point(ilon, ilat) for (ilon, ilat) in zip(lons_in_bound, lats_in_bound)]

            # mask True/False to decide whether the points are inside the glacier geometry
            # this command may be slow if many points in bound
            mask_points_in_glacier = glacier_geometry.contains(points_in_bound)

            # select only those points inside the glacier
            df_poins_in_glacier = df_points_in_bound[mask_points_in_glacier]
            lats_in_glacier = df_poins_in_glacier['POINT_LAT']
            lons_in_glacier = df_poins_in_glacier['POINT_LON']

            # if no points inside the glacier move on
            if len(df_poins_in_glacier) == 0: continue

            # if some points inside the glacier lets interpolate
            ithf_data_glacier = file_glacier_farinotti_4326.interp(y=xarray.DataArray(lats_in_glacier),
                                            x=xarray.DataArray(lons_in_glacier), method="nearest").data.squeeze()

            # plot
            ifplot = False
            p = random.randrange(0, 100, 1)
            if (ifplot and p > 0): ifplot = True
            if ifplot:
                fig, ax1 = plt.subplots()
                im1 = file_glacier_farinotti_4326.plot(ax=ax1, vmin=np.nanmin(ithf_data_glacier), vmax=np.nanmax(ithf_data_glacier), cmap='plasma')
                s1 = ax1.scatter(x=lons_in_bound, y=lats_in_bound, s=50, c='none', ec=(0, 1, 0, 1))
                s2 = ax1.scatter(x=lons_in_glacier, y=lats_in_glacier, s=50, c=ithf_data_glacier,
                                 vmin=np.nanmin(ithf_data_glacier), vmax=np.nanmax(ithf_data_glacier),
                                 ec='magenta', cmap='plasma', zorder=1)

                for n, gl in enumerate(oggm_rgi_glaciers_geoms):
                    ax1.plot(*gl.exterior.xy, c='g')
                ax1.plot(*glacier_geometry.exterior.xy, c='magenta')
                plt.show()


            # add to dataframe
            glathida.loc[df_poins_in_glacier.index, 'ith_f'] = ithf_data_glacier


        # mosaic, interpolate the mosaic and add data to dataframe
        # mosaic_ithf_4326 = merge.merge_arrays(list_glaciers_farinotti_4326, method='max')
        #ithf_data_reproj = mosaic_ithf_4326.interp(y=lats_xar, x=lons_xar, method="linear").data.squeeze()
        #glathida.loc[glathida_rgi.index, 'ith_f2'] = ithf_data_reproj

        # plot mosaic and glaciers
        plot_mosaic = False
        if plot_mosaic:
            mosaic_ithf_4326 = merge.merge_arrays(list_glaciers_farinotti_4326, method='max')
            fig, ax1 = plt.subplots()
            im1 = mosaic_ithf_4326.plot(ax=ax1, cmap='plasma')
            for n, gl in enumerate(oggm_rgi_glaciers_geoms):
                ax1.plot(*gl.exterior.xy, c='g')
            plt.show()

        # compare interpolating the mosaic vs interpolating the single rasters
        # for i in range(len(glathida_rgi)):
        #    print(glathida['ith_f'][i], '\t', glathida['ith_f2'][i])
        #print(glathida['ith_f'].describe())
        #print(glathida['ith_f2'].describe())
        #fig, ax1 = plt.subplots()
        #h1 = ax1.hist(glathida['ith_f'], bins=np.arange(0, 1300, 2), color='r', alpha=.3)
        #h2 = ax1.hist(glathida['ith_f2'], bins=np.arange(0, 1300, 2), color='b', alpha=.3)
        #plt.show()

    return glathida

if __name__ == '__main__':

    run_create_dataset = True
    if run_create_dataset:
        print(f'Begin Metadata dataset creation !')
        t0 = time.time()

        #glathida = pd.read_csv(args.path_ttt_csv, low_memory=False)
        #glathida = add_rgi(glathida, args.path_O1Regions_shp)
        #glathida = add_RGIId_and_OGGM_stats(glathida, args.OGGM_folder)
        #glathida = add_slopes_elevation(glathida, args.mosaic)
        #glathida = add_millan_vx_vy_ith(glathida, args.millan_velocity_folder, args.millan_icethickness_folder)
        #glathida = add_dist_from_boder_using_geometries(glathida)
        #glathida = add_farinotti_ith(glathida, args.farinotti_icethickness_folder)

        glathida = pd.read_csv(args.path_ttt_rgi_csv.replace('TTT_rgi.csv', 'metadata5.csv'), low_memory=False)
        #glathida = add_farinotti_ith(glathida, args.farinotti_icethickness_folder)
        #glathida = add_RGIId_and_OGGM_stats(glathida, args.OGGM_folder)
        #glathida = add_dist_from_boder_using_geometries(glathida)
        #glathida = add_slopes_elevation(glathida, args.mosaic)
        glathida = add_millan_vx_vy_ith(glathida, args.millan_velocity_folder, args.millan_icethickness_folder)

        if args.save:
            glathida.to_csv(args.save_outname, index=False)
            print(f"Metadata dataset saved: {args.save_outname}")
        print(f'Finished in {(time.time()-t0)/60} minutes. Bye bye.')
        exit()


    # Run some stuff
    glathida = pd.read_csv(args.path_ttt_csv.replace('.csv', '_final4.csv'), low_memory=False)
    pd.set_option('display.max_columns', None)
    rgis = [3, 7, 8, 11, 18]
    for rgi in rgis:

        glathida_i = glathida.loc[((glathida['RGI'] == rgi) & (glathida['SURVEY_DATE'] > 20050000))]

        print(f'{rgi} - {len(glathida_i)}')

        print(glathida_i.describe())
        """    'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax'"""
        fig, axes = plt.subplots(4,2)
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()
        ax1.hist(glathida_i['Area'], bins=100)
        ax2.hist(glathida_i['Zmin'], bins=100)
        ax3.hist(glathida_i['Zmed'], bins=100)
        ax4.hist(glathida_i['Zmax'], bins=100)
        ax5.hist(glathida_i['Slope'], bins=100)
        ax6.hist(glathida_i['Lmax'], bins=100)
        ax7.hist(glathida_i['Aspect'], bins=100)
        ax8.hist(glathida_i['TermType'], bins=100)
        plt.show()

        input('next')

        # Correlations between varibles
        # print(glathida_i.corr(method='pearson', numeric_only=True)['THICKNESS'].abs().sort_values(ascending=False))

    exit()
