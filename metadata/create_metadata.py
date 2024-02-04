import os, sys, time
import pandas as pd
from glob import glob
import random
import xarray, rioxarray
import argparse
import rasterio
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
import shapely
from shapely.geometry import Point
from matplotlib.colors import Normalize, LogNorm
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from math import radians, cos, sin, asin, sqrt
from pyproj import Transformer, CRS, Geod
import utm
"""
This program creates a dataframe of metadata for the points in glathida.

Run time evaluated on on rgi = [3,7,8,11,18]

1. add_rgi. Time: 2min.
2. add_RGIId_and_OGGM_stats. Time: 30 min.
3. add_slopes_elevation. Time: 30 min.
    - No nan can be produced here
4. add_millan_vx_vy_ith. Time: 1 min
    - Points inside the glaicer but close to the borders can be interpolated as nan.
    - Note: method to interpolate is chosen as "nearest" to reduce as much as possible these nans.
5. add_dist_from_border_in_out. Time: 6.5 h
6. add_dist_from_boder_using_geometries. Time: 6 h
7. add_farinotti_ith. Time: 45 min
    - Points inside the glaicer but close to the borders can be interpolated as nan.
    - Note: method to interpolate is chosen as "nearest" to reduce as much as possible these nans.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--path_ttt_csv', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT.csv",
                    help="Path to TTT.csv file")
parser.add_argument('--path_ttt_rgi_csv', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_rgi.csv",
                    help="Path to TTT_rgi.csv file")
parser.add_argument('--path_O1Regions_shp', type=str,default="/home/nico/OGGM/rgi/RGIV62/00_rgi62_regions/00_rgi62_O1Regions.shp",
                    help="Path to OGGM's 00_rgi62_O1Regions.shp shapefiles of all 19 RGI regions")
parser.add_argument('--mosaic', type=str,default="/media/nico/samsung_nvme/ASTERDEM_v3_mosaics/",
                    help="Path to DEM mosaics")
parser.add_argument('--millan_velocity_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/velocity/",
                    help="Path to Millan velocity data")
parser.add_argument('--millan_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/thickness/",
                    help="Path to Millan ice thickness data")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Farinotti/",
                    help="Path to Farinotti ice thickness data")
parser.add_argument('--OGGM_folder', type=str,default="/home/nico/OGGM", help="Path to OGGM main folder")
parser.add_argument('--save', type=bool, default=False, help="Save final dataset or not.")
parser.add_argument('--save_outname', type=str,
            default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_final3.csv",
            help="Saved dataframe name.")


#todo: I loop over 'GlaThiDa_ID' instead of 'RGIId'. I should see what's the difference
#todo: add dvx/dx, dvy/dy ?
#todo 1: instead of regions = [3,7,8,11,18] I'd need to compute likely glathida['RGI'].unique().tolist().
# I could but not many points to gain from the dataset.
#todo 2: Smoothing vx, vy, slopes, before interpolating ? See scipy.ndimage.convolve or some gaussian filters
#todo: add additional features from OGGM E.g. Aspect ? Or glathida ?
#todo: whenever i call clip_box i need to check if there is only 1 measurement !
#todo: change all reprojecting method from nearest to something else, like bisampling

utils.get_rgi_dir(version='62')
utils.get_rgi_intersects_dir(version='62')

args = parser.parse_args()

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def from_lat_lon_to_utm_and_epsg(lat, lon):
    """https://github.com/Turbo87/utm"""
    # Note lat lon can be also NumPy arrays.
    # In this case zone letter and number will be calculate from first entry.
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    southern_hemisphere_TrueFalse = True if zone_letter < 'N' else False
    epsg_code = 32600 + zone_number + southern_hemisphere_TrueFalse * 100
    return (easting, northing, zone_number, zone_letter, epsg_code)

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
    #all_regions = [region1, region2, region4]
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

    if (any(ele in list(glathida) for ele in ['slope_lat', 'slope_lon', 'elevation_astergdem'])):
        print('Variables slope_lat, slope_lon or elevation_astergdem already in dataframe.')
        #return glathida

    glathida['elevation_astergdem'] = [np.nan] * len(glathida)
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
    datax = []  # just to analyse the results
    datay = []  # just to analyse the results

    for rgi in [3,7,8,11,18]:

        print(f'rgi: {rgi}')
        # get dem
        dem_rgi = rioxarray.open_rasterio(path_mosaic + f'mosaic_RGI_{rgi:02d}.tif')
        ris_ang = dem_rgi.rio.resolution()[0]

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi] # collapse glathida to specific rgi
        ids_rgi = glathida_rgi['GlaThiDa_ID'].unique().tolist()  # unique IDs

        for id_rgi in tqdm(ids_rgi, total=len(ids_rgi), leave=True):

            glathida_id = glathida_rgi.loc[glathida_rgi['GlaThiDa_ID'] == id_rgi]  # collapse glathida_rgi to specific id
            glathida_id = glathida_id.copy()
            indexes_all = glathida_id.index.tolist() # useless

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

                eps = 5 * ris_ang

                # clip
                try:
                    focus = dem_rgi.rio.clip_box(
                        #minx=swlon - (deltalon + eps),
                        #miny=swlat - (deltalat + eps),
                        #maxx=nelon + (deltalon + eps),
                        #maxy=nelat + (deltalat + eps),
                        minx = swlon - delta,
                        miny = swlat - delta,
                        maxx = nelon + delta,
                        maxy = nelat + delta,
                    )
                except:
                    raise ValueError(f"Problems in method add_slopes_elevation for glacier_id: {id_rgi}")

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

                # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
                num_px_sigma_50 = max(1, round(50/res_utm_metres))
                num_px_sigma_100 = max(1, round(100/res_utm_metres))
                num_px_sigma_150 = max(1, round(150/res_utm_metres))
                num_px_sigma_300 = max(1, round(300/res_utm_metres))

                def gaussian_filter_with_nans(U, sigma):
                    # Since the reprojection into utm leads to distortions (=nans) we need to take care of this during filtering
                    # From David in https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
                    V = U.copy()
                    V[np.isnan(U)] = 0
                    VV = scipy.ndimage.gaussian_filter(V, sigma=[sigma, sigma], mode='nearest')
                    W = np.ones_like(U)
                    W[np.isnan(U)] = 0
                    WW = scipy.ndimage.gaussian_filter(W, sigma=[sigma, sigma], mode='nearest')
                    WW[WW==0]=np.nan
                    filtered_U = VV/WW
                    return filtered_U

                # Apply filter
                focus_filter_50_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_50)
                focus_filter_100_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_100)
                focus_filter_150_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_150)
                focus_filter_300_utm = gaussian_filter_with_nans(U=focus_utm_clipped.values, sigma=num_px_sigma_300)

                # Mask back the filtered arrays
                focus_filter_50_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_50_utm)
                focus_filter_100_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_100_utm)
                focus_filter_150_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_150_utm)
                focus_filter_300_utm = np.where(np.isnan(focus_utm_clipped.values), np.nan, focus_filter_300_utm)

                # create xarray object of filtered dem
                focus_filter_xarray_50_utm = focus_utm_clipped.copy(data=focus_filter_50_utm)
                focus_filter_xarray_100_utm = focus_utm_clipped.copy(data=focus_filter_100_utm)
                focus_filter_xarray_150_utm = focus_utm_clipped.copy(data=focus_filter_150_utm)
                focus_filter_xarray_300_utm = focus_utm_clipped.copy(data=focus_filter_300_utm)

                # calculate slopes for restricted dem
                # using numpy.gradient dz_dlat, dz_dlon = np.gradient(focus_utm_clipped.values, -res_utm_metres, res_utm_metres)  # [m/m]
                dz_dlat_xar, dz_dlon_xar = focus_utm_clipped.differentiate(coord='y'), focus_utm_clipped.differentiate(coord='x')
                dz_dlat_filter_xar_50, dz_dlon_filter_xar_50 = focus_filter_xarray_50_utm.differentiate(coord='y'), focus_filter_xarray_50_utm.differentiate(coord='x')
                dz_dlat_filter_xar_100, dz_dlon_filter_xar_100 = focus_filter_xarray_100_utm.differentiate(coord='y'), focus_filter_xarray_100_utm.differentiate(coord='x')
                dz_dlat_filter_xar_150, dz_dlon_filter_xar_150 = focus_filter_xarray_150_utm.differentiate(coord='y'), focus_filter_xarray_150_utm.differentiate(coord='x')
                dz_dlat_filter_xar_300, dz_dlon_filter_xar_300  = focus_filter_xarray_300_utm.differentiate(coord='y'), focus_filter_xarray_300_utm.differentiate(coord='x')

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

                # check if any nan in the interpolate data
                contains_nan = any(np.isnan(arr).any() for arr in [slope_lon_data, slope_lat_data,
                                                                   slope_lon_data_filter_50, slope_lat_data_filter_50,
                                                                   slope_lon_data_filter_100, slope_lat_data_filter_100,
                                                                   slope_lon_data_filter_150, slope_lat_data_filter_150,
                                                                   slope_lon_data_filter_300, slope_lat_data_filter_300])
                if contains_nan:
                    raise ValueError(f"Nan detected in elevation/slope calc. Check")

                # other checks
                assert slope_lat_data.shape == slope_lon_data.shape == elevation_data.shape, "Different shapes, something wrong!"
                assert slope_lat_data_filter_150.shape == slope_lon_data_filter_150.shape == elevation_data.shape, "Different shapes, something wrong!"
                assert len(slope_lat_data) == len(indexes_all_epsg), "Different shapes, something wrong!"

                # write to dataframe
                glathida.loc[indexes_all_epsg, 'elevation_astergdem'] = elevation_data
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
            glathida.loc[indexes_all, 'elevation_astergdem'] = elevation_data
            glathida.loc[indexes_all, 'slope_lat_gf50'] = slope_lat_data_filter_50
            glathida.loc[indexes_all, 'slope_lon_gf50'] = slope_lon_data_filter_50
            glathida.loc[indexes_all, 'slope_lat_gf100'] = slope_lat_data_filter_100
            glathida.loc[indexes_all, 'slope_lon_gf100'] = slope_lon_data_filter_100
            glathida.loc[indexes_all, 'slope_lat_gf150'] = slope_lat_data_filter_150
            glathida.loc[indexes_all, 'slope_lon_gf150'] = slope_lon_data_filter_150
            glathida.loc[indexes_all, 'slope_lat_gf300'] = slope_lat_data_filter_300
            glathida.loc[indexes_all, 'slope_lon_gf300'] = slope_lon_data_filter_300
            """
    return glathida

"""Add Millan's velocity vx, vy, ith"""
def add_millan_vx_vy_ith(glathida, path_millan_velocity, path_millan_icethickness):
    """
    :param glathida: input csv. dataframe to which I want to add velocity
    :param path_millan_velocity: path to folder with millan velocity data
    :return: dataframe with velocity added
    """
    print('Adding Millan velocity and ice thickness method...')
    #todo: smussare con filtro gaussiano oppure mediare un intorno del punto per vx, vy ?
    # todo: Millan usa filtro mediano + filtro gaussiano con width variabili

    if (any(ele in list(glathida) for ele in ['vx', 'vy', 'ith_m'])):
        print('Variable already in dataframe.')
        return glathida

    glathida['vx'] = [np.nan] * len(glathida)
    glathida['vy'] = [np.nan] * len(glathida)
    glathida['ith_m'] = [np.nan] * len(glathida)

    for rgi in [3, 7, 8, 11, 18]:

        glathida_rgi = glathida.loc[glathida['RGI'] == rgi] # glathida to specific rgi
        print(f'rgi: {rgi}, Total: {len(glathida_rgi)}')
        ids_rgi = glathida_rgi['GlaThiDa_ID'].unique().tolist()  # unique IDs

        # get Millan vx files for specific rgi
        files_vx = glob(path_millan_velocity+'RGI-{}/VX_RGI-{}*'.format(rgi, rgi))
        # merge vx files to create vx mosaic
        xds_4326 = []
        for tiffile in files_vx:
            xds = rioxarray.open_rasterio(tiffile, masked=False)
            xds.rio.write_nodata(np.nan, inplace=True)
            xds_4326.append(xds)#xds.rio.reproject("EPSG:4326"))
        mosaic_vx = merge.merge_arrays(xds_4326)

        # get Millan vy files for specific rgi
        files_vy = glob(args.millan_velocity_folder+'RGI-{}/VY_RGI-{}*'.format(rgi, rgi))
        # merge vy files to create vy mosaic
        xds_4326 = []
        for tiffile in files_vy:
            xds = rioxarray.open_rasterio(tiffile, masked=False)
            xds.rio.write_nodata(np.nan, inplace=True)
            xds_4326.append(xds)#xds.rio.reproject("EPSG:4326"))
        mosaic_vy = merge.merge_arrays(xds_4326)

        # get Millan ice thickness files for specific rgi
        files_ith = glob(path_millan_icethickness+'RGI-{}/THICKNESS_RGI-{}*'.format(rgi, rgi))
        # merge ice thickness files to create mosaic
        xds_4326 = []
        for tiffile in files_ith:
            xds = rioxarray.open_rasterio(tiffile, masked=False)
            xds.rio.write_nodata(np.nan, inplace=True)
            xds_4326.append(xds)#xds.rio.reproject("EPSG:4326"))
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
            #print(new_llx, new_lly, new_urx, new_ury)
            mosaic_ith = mosaic_ith.rio.clip_box(minx=new_llx, miny=new_lly, maxx=new_urx, maxy=new_ury)
            mosaic_vx = mosaic_vx.rio.clip_box(minx=new_llx, miny=new_lly, maxx=new_urx, maxy=new_ury)
            mosaic_vy = mosaic_vy.rio.clip_box(minx=new_llx, miny=new_lly, maxx=new_urx, maxy=new_ury)
            #for i in [mosaic_ith, mosaic_vx, mosaic_vy]:
            #    print(i.rio.width, i.rio.height, i.rio.bounds(), i.rio.crs, i.rio.resolution()[0])

        # IMPORTANT
        # velocity maps contains nans, while the ice thickness map does not. This may result in nans in the
        # velocity interpolation, while non-nans in the ice thickness interpolation.
        # I decide therefore to mask the ice thickness data based-on velocity so that if nan results for
        # interpolating the velocity, also nan results when interpolating the ice thickness
        mosaic_ith = mosaic_ith.where(mosaic_vx.notnull())


        # Note ris_ang for lat/lon/ith mosaics may be slightly different even for the same rgi, but we don't care here
        ris_ang = mosaic_vx.rio.resolution()[0]
        crs = mosaic_vx.rio.crs
        eps = 5 * ris_ang
        transformer = Transformer.from_crs("EPSG:4326", crs)

        # loop over the unique IDs
        for id_rgi in tqdm(ids_rgi, total=len(ids_rgi), leave=True):

            glathida_id = glathida_rgi.loc[glathida_rgi['GlaThiDa_ID'] == id_rgi] #  glathida_rgi to specific id
            indexes_all = glathida_id.index.tolist()

            lats = np.array(glathida_id['POINT_LAT'])
            lons = np.array(glathida_id['POINT_LON'])

            lons, lats = transformer.transform(lats, lons)

            swlat = np.amin(lats)
            swlon = np.amin(lons)
            nelat = np.amax(lats)
            nelon = np.amax(lons)

            deltalat = np.abs(swlat - nelat)
            deltalon = np.abs(swlon - nelon)
            lats_xar = xarray.DataArray(lats)
            lons_xar = xarray.DataArray(lons)

            # clip millan mosaic
            try:
                focus_vx = mosaic_vx.rio.clip_box(minx=swlon - deltalon - eps, miny=swlat - deltalat - eps,
                                                    maxx=nelon + deltalon + eps, maxy=nelat + deltalat + eps)

                focus_vy = mosaic_vy.rio.clip_box(minx=swlon - deltalon - eps, miny=swlat - deltalat - eps,
                                                    maxx=nelon + deltalon + eps, maxy=nelat + deltalat + eps)

                focus_ith = mosaic_ith.rio.clip_box(minx=swlon - deltalon - eps, miny=swlat - deltalat - eps,
                                                    maxx=nelon + deltalon + eps, maxy=nelat + deltalat + eps)

            except:
                tqdm.write(f'non ci sono dati di millan per rgi {rgi} glacier_id: {id_rgi} ')
                #print(swlon - deltalon - eps, nelon + deltalon + eps, swlat - deltalat - eps, nelat + deltalat + eps)
                continue


            # interpolate
            vx_data = focus_vx.interp(y=lats_xar, x=lons_xar, method="nearest").data.squeeze()
            vy_data = focus_vy.interp(y=lats_xar, x=lons_xar, method="nearest").data.squeeze()
            ith_data = focus_ith.interp(y=lats_xar, x=lons_xar, method="nearest").data.squeeze()
            #print(np.sum(np.isnan(vx_data)), np.sum(np.isnan(vy_data)), np.sum(np.isnan(ith_data)))

            # Define a smoothing kernel (e.g., a nxn box filter)
            #kernel = np.ones((3, 3)) / 9
            #conv2d = lambda x: scipy.ndimage.convolve(x, kernel, mode="nearest")
            #gaus = lambda x: scipy.ndimage.gaussian_filter(x, sigma=1., mode="nearest")
            # Apply convolution to smooth the xarray
            #smth_vx = xarray.apply_ufunc(conv2d, focus_vx[0,:,:])
            #smth_vy = xarray.apply_ufunc(conv2d, focus_vy[0,:,:])
            #smth_ith = xarray.apply_ufunc(conv2d, focus_ith[0,:,:])

            # plot
            ifplot = False
            if ifplot and np.any(np.isnan(vx_data)): ifplot = True
            if ifplot:
                fig, axes = plt.subplots(1,3, figsize=(10,3))
                ax1, ax2, ax3 = axes.flatten()

                im1 = focus_vx.plot(ax=ax1, cmap='viridis', vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data))
                s1 = ax1.scatter(x=lons, y=lats, s=15, c=vx_data, ec=(0,0,0,0.1), cmap='viridis',
                                 vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data), zorder=1)
                s1_1 = ax1.scatter(x=lons[np.argwhere(np.isnan(vx_data))], y=lats[np.argwhere(np.isnan(vx_data))], s=15,
                                   c='magenta', zorder=1)

                im2 = focus_vy.plot(ax=ax2, cmap='viridis', vmin=np.nanmin(vy_data), vmax=np.nanmax(vy_data))
                s2 = ax2.scatter(x=lons, y=lats, s=15, c=vy_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                 vmin=np.nanmin(vy_data), vmax=np.nanmax(vy_data), zorder=1)
                s2_1 = ax2.scatter(x=lons[np.argwhere(np.isnan(vy_data))], y=lats[np.argwhere(np.isnan(vy_data))], s=15,
                                   c='magenta', zorder=1)

                im3 = focus_ith.plot(ax=ax3, cmap='viridis', vmin=0, vmax=np.nanmax(ith_data))
                s3 = ax3.scatter(x=lons, y=lats, s=15, c=ith_data, ec=(0, 0, 0, 0.1), cmap='viridis',
                                 vmin=0, vmax=np.nanmax(ith_data), zorder=1)
                s3_1 = ax3.scatter(x=lons[np.argwhere(np.isnan(ith_data))], y=lats[np.argwhere(np.isnan(ith_data))], s=15,
                                   c='magenta', zorder=1)

                #im4 = smth_vx.plot(ax=ax4, cmap='viridis', vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data))
                #im5 = smth_vx2.plot(ax=ax5, cmap='viridis', vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data))
                #im6 = smth_ith.plot(ax=ax6, cmap='viridis', vmin=0, vmax=np.nanmax(ith_data))
                # for i, val in enumerate(ith_data): ax3.annotate(f'{val:.2f}', (lons[i], lats[i]))

                ax1.title.set_text('vx')
                ax2.title.set_text('vy')
                ax3.title.set_text('ice thickness')
                for ax in (ax1, ax2, ax3):
                    ax.set(xlabel='', ylabel='')
                plt.tight_layout()
                plt.show()

            # add vx, vy data to dataframe
            glathida.loc[indexes_all, 'vx'] = vx_data
            glathida.loc[indexes_all, 'vy'] = vy_data
            glathida.loc[indexes_all, 'ith_m'] = ith_data

        # print something to check
        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]
        print('Total:', len(glathida_rgi), ', nans in vx/vy/ith: ', np.sum(np.isnan(glathida_rgi['vx'])),
                       np.sum(np.isnan(glathida_rgi['vy'])), np.sum(np.isnan(glathida_rgi['ith_m'])))
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
    # Compared to the Millan method these distances are lower, at times much lower. Geometries contain nunataks,
    # while probably the resolution of Millan product does not see small nunataks. This algorithm consider nunataks.
    # I suspect this algorithm is more accurate than the method that uses Millan's product.
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

    regions = [3, 7, 8, 11, 18]

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

                # Create a geoseries of all external and internal geometries
                geoseries_geometries_epsg = gpd.GeoSeries(cluster_exterior_ring + cluster_interior_rings)

            # Get all points and create Geopandas geoseries and convert to glacier center UTM
            # Note: a delicate issue is that technically each point may have its own UTM zone.
            # To keep things consistent I decide to project all of them to the glacier center UTM.
            lats = glathida_id['POINT_LAT'].tolist()
            lons = glathida_id['POINT_LON'].tolist()
            list_points = [Point(lon, lat) for (lon, lat) in zip(lons, lats)]
            geoseries_points_4326 = gpd.GeoSeries(list_points, crs="EPSG:4326")
            geoseries_points_epsg = geoseries_points_4326.to_crs(epsg=glacier_epsg)

            # Finally ready to loop over the points inside glacier_id
            glacier_id_dist = []
            for i, (idx, lon, lat) in tqdm(enumerate(zip(glathida_id.index, lons, lats)), total=len(lons), desc='Points', leave=False):

                make_check0 = False
                if make_check0 and i==0:
                    lat_check = glathida_id.loc[idx, 'POINT_LAT']
                    lon_check = glathida_id.loc[idx, 'POINT_LON']
                    #print(lon_check, lat_check, lon, lat)

                # Make two checks.
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
                        print(f"Note differet UTM zones. Point espg {epsg} and glacier center epsg {glacier_epsg}.")
                        #todo: maybe need to correct for this.

                # Decide whether point is inside a nunatak. If yes set the distance to nan
                is_nunatak = any(nunatak.contains(Point(lon, lat)) for nunatak in gl_geom_nunataks_list)
                if is_nunatak is True:
                    min_dist = np.nan

                # if not
                else:
                    # get shapely Point
                    point_epsg = geoseries_points_epsg.iloc[i]
                    # Calculate the distances between such point and all glacier geometries
                    min_distances_point_geometries = geoseries_geometries_epsg.distance(point_epsg)
                    min_dist = np.min(min_distances_point_geometries)  # unit UTM: m

                    # To debug we want to check what point corresponds to the minimum distance.
                    debug_distance = False
                    if debug_distance:
                        min_distance_index = min_distances_point_geometries.idxmin()
                        nearest_line = geoseries_geometries_epsg.loc[min_distance_index]
                        nearest_point_on_line = nearest_line.interpolate(nearest_line.project(point_epsg))
                        # print(f"{i} Minimum distance: {min_dist:.2f} meters.")

                # Fill distance list for glacier id of point
                glacier_id_dist.append(min_dist/1000.)

                # Compare with Millan's distance method
                check_with_millan = False
                if check_with_millan:
                    dist_with_millan = glathida_id.loc[idx, 'dist_from_border_km']
                    if (abs(min_dist/1000.-dist_with_millan)/(min_dist/1000.)>2.):
                        millan_mismatch=True
                        print(f"Geometry method: {min_dist / 1000.:.5f} Millan method: {dist_with_millan:.5f}")

                # Plot
                plot_calculate_distance = False
                r = random.uniform(0,1)
                if (plot_calculate_distance and list_cluster_RGIIds is not None and r<1.0):
                    plot_calculate_distance = True
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

                    if is_nunatak: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='b')
                    else: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='r', ec='r')
                    if debug_distance: ax2.scatter(*nearest_point_on_line.xy, s=50, lw=2, c='g')

                    ax1.set_title('EPSG 4326')
                    ax2.set_title(f'EPSG {glacier_epsg}')
                    plt.show()


            assert len(glacier_id_dist)==len(glathida_id.index), "Number of measurements does not match index length"
            tqdm.write(f"Finished glacier {rgi_id} with {len(glacier_id_dist)} measurements")

            # Fill dataframe with distances from glacier rgi_id
            glathida.loc[glathida_id.index, 'dist_from_border_km_geom'] = glacier_id_dist

        tqdm.write(f"Finished region {rgi} in {(time.time()-t_)/60} mins.")

    return glathida

"""Add RGIId and other OGGM stats like glacier area"""
def add_RGIId_and_OGGM_stats(glathida, path_OGGM_folder):
    # Note: points that are outside any glaciers will have nan to RGIId (and the other features)

    print("Adding OGGM's stats method...")
    if (any(ele in list(glathida) for ele in ['RGIId', 'Area'])):
        print('Variables RGIId/Area already in dataframe.')
        return glathida

    glathida['RGIId'] = [np.nan] * len(glathida)
    glathida['Area'] = [np.nan] * len(glathida)
    glathida['Zmin'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Zmax'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Zmed'] = [np.nan] * len(glathida) # -999 are bad values
    glathida['Slope'] = [np.nan] * len(glathida)
    glathida['Lmax'] = [np.nan] * len(glathida) # -9 missing values, see https://essd.copernicus.org/articles/14/3889/2022/essd-14-3889-2022.pdf
    #todo: have I managed these bad values ? -999 and -9 ?
    #todo: add Athe following features: Form, TermType, Status, Aspect
    # todo: check if any nan results from this method

    regions = [3, 7, 8, 11, 18]

    for rgi in tqdm(regions, total=len(regions), leave=True):
        # get OGGM's dataframe of rgi glaciers
        oggm_rgi_shp = glob(f'{path_OGGM_folder}/rgi/RGIV62/{rgi:02d}*/{rgi:02d}*.shp')[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
        #print(oggm_rgi_glaciers.describe())
        #input('wait')
        #print(oggm_rgi_glaciers['Status'].unique())
        #print(oggm_rgi_glaciers['TermType'].unique())
        #print(oggm_rgi_glaciers['Surging'].unique())
        #print(oggm_rgi_glaciers.head(5))
        print(f'rgi: {rgi}. Imported OGGMs {oggm_rgi_shp} dataframe of {len(oggm_rgi_glaciers)} glaciers')

        # Glathida
        glathida_rgi = glathida.loc[glathida['RGI'] == rgi]

        # loop sui ghiacciai di oggm
        for ind in tqdm(oggm_rgi_glaciers.index, total=len(oggm_rgi_glaciers)):

            glacier_geometry = oggm_rgi_glaciers.loc[ind, 'geometry']
            glacier_RGIId = oggm_rgi_glaciers.loc[ind, 'RGIId']
            glacier_area = oggm_rgi_glaciers.loc[ind, 'Area'] #km2
            glacier_zmin = oggm_rgi_glaciers.loc[ind, 'Zmin']
            glacier_zmax = oggm_rgi_glaciers.loc[ind, 'Zmax']
            glacier_zmed = oggm_rgi_glaciers.loc[ind, 'Zmed']
            glacier_slope = oggm_rgi_glaciers.loc[ind, 'Slope']
            glacier_lmax = oggm_rgi_glaciers.loc[ind, 'Lmax']
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
            mask_points_in_glacier = glacier_geometry.contains(points_in_bound)

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

            # Add to dataframe
            glathida.loc[df_poins_in_glacier.index, 'RGIId'] = glacier_RGIId
            glathida.loc[df_poins_in_glacier.index, 'Area'] = glacier_area
            glathida.loc[df_poins_in_glacier.index, 'Zmin'] = glacier_zmin
            glathida.loc[df_poins_in_glacier.index, 'Zmax'] = glacier_zmax
            glathida.loc[df_poins_in_glacier.index, 'Zmed'] = glacier_zmed
            glathida.loc[df_poins_in_glacier.index, 'Slope'] = glacier_slope
            glathida.loc[df_poins_in_glacier.index, 'Lmax'] = glacier_lmax

    return glathida

"""Add Farinotti's ith"""
def add_farinotti_ith(glathida, path_farinotti_icethickness):
    print("Adding Farinotti ice thickness...")

    if ('ith_f' in list(glathida)):
        print('Variable already in dataframe.')
        return glathida

    glathida['ith_f'] = [np.nan] * len(glathida)

    regions = [3,7,8,11,18]

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
        files_names_farinotti = glob(path_farinotti_icethickness + f'composite_thickness_RGI60-{rgi:02d}/RGI60-{rgi:02d}/*')
        list_glaciers_farinotti_4326 = []

        for n, tiffile in tqdm(enumerate(files_names_farinotti), total=len(files_names_farinotti), leave=True):

            glacier_name = tiffile.split('/')[-1].replace('_thickness.tif', '')
            glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name, 'geometry'].item()

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
        glathida = pd.read_csv(args.path_ttt_csv.replace('.csv', '_final2.csv'), low_memory=False)
        #glathida = pd.read_csv(args.path_ttt_csv.replace('.csv', '_rgi.csv'), low_memory=False)
        glathida = add_rgi(glathida, args.path_O1Regions_shp)
        #glathida.to_csv(args.path_ttt_csv.replace('.csv', '_rgi.csv'), index=False)
        glathida = add_RGIId_and_OGGM_stats(glathida, args.OGGM_folder)
        glathida = add_slopes_elevation(glathida, args.mosaic)
        glathida = add_millan_vx_vy_ith(glathida, args.millan_velocity_folder, args.millan_icethickness_folder)
        glathida = add_dist_from_border_in_out(glathida, args.millan_velocity_folder)
        glathida = add_dist_from_boder_using_geometries(glathida)
        glathida = add_farinotti_ith(glathida, args.farinotti_icethickness_folder)

        if args.save:
            glathida.to_csv(args.save_outname, index=False)
            print('Metadata dataset saved.')
        print(f'Finished in {(time.time()-t0)/60} minutes. Bye bye.')
        exit()


    # Run some stuff
    glathida = pd.read_csv(args.path_ttt_csv.replace('.csv', '_final.csv'), low_memory=False)
    glathida2 = pd.read_csv(args.path_ttt_csv.replace('.csv', '_final3.csv'), low_memory=False)
    pd.set_option('display.max_columns', None)
    rgis = [3, 7, 8, 11]
    for rgi in rgis:

        glathida_i = glathida.loc[((glathida['RGI'] == rgi) & (glathida['SURVEY_DATE'] > 20050000))]
        glathida2_i = glathida2.loc[((glathida2['RGI'] == rgi) & (glathida2['SURVEY_DATE'] > 20050000))]

        glathida2_i = glathida2_i.copy()

        glathida2_i['v'] = np.sqrt(glathida2_i['vx']**2 + glathida2_i['vy']**2)
        glathida2_i['slope'] = np.sqrt(glathida2_i['slope_lat'] ** 2 + glathida2_i['slope_lon'] ** 2)
        glathida['elevation_from_zmin'] = glathida['elevation_astergdem'] - glathida['Zmin']

        print(f'{rgi} - {len(glathida_i)} - {len(glathida2_i)}')

        # Correlations between varibles
        print(glathida2_i.corr(method='pearson', numeric_only=True)['THICKNESS'].abs().sort_values(ascending=False))

        exit()


        """ Distributions of all variables"""
        #min_ = min(glathida_i['vx'].min(), glathida_i['vx'].min())
        #max_ = max(glathida_i['vx'].max(), glathida_i['vy'].max())
        #min_ = glathida_i['elevation_astergdem'].min()
        #max_ = glathida_i['elevation_astergdem'].max()
        #print(min_, max_)
        #v = max(abs(min_), abs(max_))+1
        #fig, ax = plt.subplots()
        #h1 = plt.hist(glathida_i['elevation_astergdem'], color='r', alpha=0.5, bins=np.arange(0, v, 10), log=False, label='elevation_astergdem')
        #h1 = plt.hist(glathida_i['vx'], color='r', alpha=0.5, bins=np.arange(-v, v, 1), log=True, label='vx')
        #h2 = plt.hist(glathida_i['vy'], color='b', alpha=0.5, bins=np.arange(-v, v, 1), log=True, label='vy')
        #plt.legend(title=f'elevation_astergdem - rgi{rgi}')
        #plt.show()


        """
        Visualize scatter plot between 2 variables + 1 as color 
        """
        """
        glathida_i.sort_values('THICKNESS', inplace=True)

        fig, ax1  = plt.subplots(1,1, figsize=(8,8))
        #s1 = ax1.scatter(x=glathida_i['slope_lon'], y=glathida_i['vx'], c=glathida_i['THICKNESS'],
        #                 s=1, alpha=1, norm=None, cmap='plasma', label=f'rgi: {rgi}')
        #s2 = ax2.scatter(x=glathida_i['slope_lat'], y=glathida_i['vy'], c=glathida_i['THICKNESS'],
        #                 s=1, alpha=1, norm=None, cmap='plasma', label=f'rgi: {rgi}')

        s1 = ax1.scatter(x=glathida_i['dist_from_border_km'], y=glathida_i['elevation_astergdem'], c=glathida_i['THICKNESS'],
                         s=1, alpha=1, norm=None, cmap='plasma', label=f'rgi: {rgi}')

        for ax in (ax1, ):
            ax.axhline(y=0, color='grey', alpha=.3)
            ax.axvline(x=0, color='grey', alpha=.3)
            ax.legend()
        ax1.set_xlabel('dist_from_border_km')
        ax1.set_ylabel('elevation_astergdem')
        #ax2.set_xlabel('slope_lat')
        #ax2.set_ylabel('vy (m/yr)')
        cbar1 = plt.colorbar(s1, ax=ax1, alpha=1)
        #cbar2 = plt.colorbar(s2, ax=ax2, alpha=1)
        for cbar in (cbar1, ): cbar.set_label('THICKNESS (m)', labelpad=15, rotation=270)
        plt.tight_layout()
        plt.show()"""

        """
        Glathida Ice thickness vs all variables
        """
        """
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(19, 5))
        s1 = ax1.scatter(x=glathida_i['slope_lon'], y=glathida_i['THICKNESS'], s=1)
        s2 = ax2.scatter(x=glathida_i['slope_lat'], y=glathida_i['THICKNESS'], s=1)
        s3 = ax3.scatter(x=glathida_i['vx'], y=glathida_i['THICKNESS'], s=1)
        s4 = ax4.scatter(x=glathida_i['vy'], y=glathida_i['THICKNESS'], s=1)
        s5 = ax5.scatter(x=glathida_i['dist_from_border_km'], y=glathida_i['THICKNESS'], s=1)

        xtitles = ['slope_lon', 'slope_lat', 'vx', 'vy', 'dist_from_border']
        for i in range(len(fig.axes)):
            fig.axes[i].set_xlabel(xtitles[i])
            fig.axes[i].set_ylabel('Ice thickness (m)')

        fig.suptitle(f'rgi {rgi}', fontsize=16)
        fig.tight_layout(pad=1.0)
        plt.show()"""
        """
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        s1 = ax.scatter(x=glathida_i['elevation_astergdem'], y=glathida_i['THICKNESS'], s=1)
        ax.set_xlabel('elevation_astergdem')
        ax.set_ylabel('Ice thickness (m)')
        fig.suptitle(f'rgi {rgi}', fontsize=16)
        fig.tight_layout(pad=1.0)
        plt.show()"""

        """
        Glathida Ice thickness vs Millan ice thickness
        """
        """
        diff = glathida_i['THICKNESS'] - glathida_i['ith_m']

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
        s1 = ax1.scatter(x=glathida_i['THICKNESS'], y=glathida_i['ith_m'], s=1)
        lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]), np.max([ax1.get_xlim(), ax1.get_ylim()])]
        ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=1)
        h1 = ax2.hist(diff, bins=np.arange(diff.min(), diff.max(), 1))

        ax2.text(0.6, 0.9, f'rgi: {rgi}', transform=ax2.transAxes)
        ax2.text(0.6, 0.85, f'mean = {diff.mean():.1f} m', transform=ax2.transAxes)
        ax2.text(0.6, 0.8, f'median = {diff.median():.1f} m', transform=ax2.transAxes)
        ax2.text(0.6, 0.75, f'std = {diff.std():.1f} m', transform=ax2.transAxes)

        ax1.set_xlabel('Glathida ice thickness (m)')
        ax1.set_ylabel('Millan ice thickness (m)')
        ax2.set_xlabel('Glathida - Millan ice thick (m)')
        ax2.set_ylabel('Counts')

        plt.tight_layout()
        plt.show()"""








