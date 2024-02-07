import os
import sys
sys.path.append("/home/nico/PycharmProjects/skynet/code") # to import haversine from utils.py
from utils import haversine
from glob import glob
import argparse
import numpy as np
import xarray
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import rioxarray
import rasterio
from rioxarray import merge
import geopandas as gpd
import oggm
from oggm import utils
from shapely.geometry import Point, Polygon
from pyproj import Proj, Transformer
from math import radians, cos, sin, asin, sqrt, floor
import utm

"""
This program generates glacier metadata at some random locations inside the glacier geometry. 

Input: glacier name (RGIId), how many points you want to generate. 
Output: pandas dataframe with features calculated for each generated point. 

Note: the features slope, elevation_from_zmin and v are calculated in model.py, not here.

Note: the points are generated inside the glacier but outside nunataks (there is a check for this)

Note: Millan and Farinotti products needs to be interpolated. Interpolation close to the borders may result in nans. 
The interpolation method="nearest" yields much less nans close to borders if compared to linear
interpolation and therefore is chosen. 

Note that Farinotti interpolation ith_f may result in nan when generated point too close to the border.

Note the following policy for Millan special cases to produce vx, vy, v, ith_m:
    1) There is no Millan data for such glacier. Data imputation: vy=vy=v=0.0 and ith_m=nan. 
    2) In case the interpolation of Millan's fields yields nan because points are either too close to the margins. 
    I keep the nans that will be however removed before returning the dataset.   
"""
# todo: smooth millan, farinotti and slope fiels before interpolation
# todo: Data imputation: Millan and other features
# todo: inserire anche un ulteriore feature che è la velocità media di tutto il ghiacciao ? sia vxm, vym, vm ?
# todo: inserire dvx/dx, dvx/dy, dvy/dx, dvy/vy ?
# todo: inserire anche la curvatura ? Vedi la tesi di farinotti, pare la curvatura sia importante
# todo: a proposito di come smussare i campi di slope e velocita, guardare questo articolo:
#  Slope estimation influences on ice thickness inversion models: a case study for Monte Tronador glaciers, North Patagonian Andes

parser = argparse.ArgumentParser()
parser.add_argument('--mosaic', type=str,default="/media/nico/samsung_nvme/ASTERDEM_v3_mosaics/",
                    help="Path to DEM mosaics")
parser.add_argument('--millan_velocity_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/velocity/",
                    help="Path to Millan velocity data")
parser.add_argument('--millan_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Millan/thickness/",
                    help="Path to Millan ice thickness data")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Farinotti/",
                    help="Path to Farinotti ice thickness data")

args = parser.parse_args()

utils.get_rgi_dir(version='62')  # setup oggm version
utils.get_rgi_intersects_dir(version='62')


def from_lat_lon_to_utm_and_epsg(lat, lon):
    """https://github.com/Turbo87/utm"""
    # Note lat lon can be also NumPy arrays.
    # In this case zone letter and number will be calculate from first entry.
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    southern_hemisphere_TrueFalse = True if zone_letter < 'N' else False
    epsg_code = 32600 + zone_number + southern_hemisphere_TrueFalse * 100
    return (easting, northing, zone_number, zone_letter, epsg_code)

def gaussian_filter_with_nans(U, sigma):
    # Since the reprojection into utm leads to distortions (=nans) we need to take care of this during filtering
    # From David in https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=[sigma, sigma], mode='nearest')
    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=[sigma, sigma], mode='nearest')
    WW[WW == 0] = np.nan
    filtered_U = VV / WW
    return filtered_U


def populate_glacier_with_metadata(glacier_name, n=50):
    print(f"******* FETCHING FEATURES FOR GLACIER {glacier_name} *******")

    rgi = int(glacier_name[6:8]) # get rgi from the glacier code
    oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62') # get rgi region shp
    oggm_rgi_intersects_shp = utils.get_rgi_intersects_region_file(f"{rgi:02d}", version='62') # get rgi intersect shp file

    oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)             # get rgi dataset of glaciers
    oggm_rgi_intersects =  gpd.read_file(oggm_rgi_intersects_shp) # get rgi dataset of glaciers intersects

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

    # Get glacier dataset
    try:
        gl_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name]
        print(f"Glacier {glacier_name} found.")
        assert len(gl_df) == 1, "Check this please."
        # print(gl_df.T)
    except Exception as e:
        print(f"Error {e}")
        exit()

    # intersects of glacier (need only for plotting purposes)
    gl_intersects = oggm.utils.get_rgi_intersects_entities([glacier_name], version='62')

    # Calculate intersects of all glaciers in the cluster
    list_cluster_RGIIds = find_cluster_RGIIds(glacier_name, oggm_rgi_intersects)
    # print(f"List of glacier cluster: {list_cluster_RGIIds}")

    if list_cluster_RGIIds is not None:
        cluster_intersects = oggm.utils.get_rgi_intersects_entities(list_cluster_RGIIds, version='62') # (need only for plotting purposes)
    else: cluster_intersects = None
    gl_geom = gl_df['geometry'].item() # glacier geometry Polygon
    gl_geom_ext = Polygon(gl_geom.exterior)  # glacier geometry Polygon
    gl_geom_nunataks_list = [Polygon(nunatak) for nunatak in gl_geom.interiors] # list of nunataks Polygons
    llx, lly, urx, ury = gl_geom.bounds # geometry bounds

    # Generate points (no points can be generated inside nunataks)
    points = {'lons': [], 'lats': [], 'nunataks': []}
    while (len(points['lons']) < n):
        r_lon = np.random.uniform(llx, urx)
        r_lat = np.random.uniform(lly, ury)
        point = Point(r_lon, r_lat)

        is_inside = gl_geom_ext.contains(point)
        is_nunatak = any(nunatak.contains(point) for nunatak in gl_geom_nunataks_list)

        if (is_inside is False or is_nunatak is True): # if outside geometry or inside any nunatak discard the point
            continue

        points['lons'].append(r_lon)
        points['lats'].append(r_lat)
        points['nunataks'].append(int(is_nunatak))

    # Feature dataframe
    points_df = pd.DataFrame(columns=['lons', 'lats', 'nunataks'])
    # Fill lats, lons and nunataks
    points_df['lats'] = points['lats']
    points_df['lons'] = points['lons']
    points_df['nunataks'] = points['nunataks']
    if (points_df['nunataks'].sum() != 0):
        print(f"The generation pipeline has produced n. {points_df['nunataks'].sum()} points inside nunataks")
        raise ValueError

    # Fill these features
    points_df['RGI'] = rgi
    points_df['Area'] = gl_df['Area'].item()
    points_df['Zmin'] = gl_df['Zmin'].item()
    points_df['Zmax'] = gl_df['Zmax'].item()
    points_df['Zmed'] = gl_df['Zmed'].item()
    points_df['Slope'] = gl_df['Slope'].item()
    points_df['Lmax'] = gl_df['Lmax'].item()
    points_df['Form'] = gl_df['Form'].item()
    points_df['TermType'] = gl_df['TermType'].item()
    points_df['Aspect'] = gl_df['Aspect'].item()

    """ Add Slopes and Elevation """
    print(f"Calculating slopes and elevations...")
    dem_rgi = rioxarray.open_rasterio(args.mosaic + f'mosaic_RGI_{rgi:02d}.tif')
    ris_ang = dem_rgi.rio.resolution()[0]

    cenLon, cenLat = gl_df['CenLon'].item(), gl_df['CenLat'].item()
    _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(cenLat, cenLon)

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
        raise ValueError(f"Problems in method for fetching add_slopes_elevation")

    focus = focus.squeeze()

    # ***************** Calculate elevation and slopes in UTM ********************
    # Reproject to utm (projection distortions along boundaries converted to nans)
    # Default resampling is nearest which leads to weird artifacts. Options are bilinear (long) and cubic (very long)
    focus_utm = focus.rio.reproject(glacier_epsg, resampling=rasterio.enums.Resampling.bilinear, nodata=-9999) # long!
    focus_utm = focus_utm.where(focus_utm != -9999, np.nan)
    # Calculate the resolution in meters of the utm focus (resolutions in x and y are the same!)
    res_utm_metres = focus_utm.rio.resolution()[0]

    eastings, northings, _, _, _ = from_lat_lon_to_utm_and_epsg(np.array(points_df['lats']),
                                                                np.array(points_df['lons']))
    northings_xar = xarray.DataArray(northings)
    eastings_xar = xarray.DataArray(eastings)

    # clip the utm with a buffer of 2 km in both dimentions. This is necessary since smoothing is otherwise long
    focus_utm = focus_utm.rio.clip_box(
        minx=min(eastings) - 2000,
        miny=min(northings) - 2000,
        maxx=max(eastings) + 2000,
        maxy=max(northings) + 2000)


    num_px_sigma_50 = max(1, round(50 / res_utm_metres))
    num_px_sigma_100 = max(1, round(100 / res_utm_metres))
    num_px_sigma_150 = max(1, round(150 / res_utm_metres))
    num_px_sigma_300 = max(1, round(300 / res_utm_metres))

    # Apply filter (utm here)
    focus_filter_50_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_50)
    focus_filter_100_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_100)
    focus_filter_150_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_150)
    focus_filter_300_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_300)

    # Mask back the filtered arrays
    focus_filter_50_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_50_utm)
    focus_filter_100_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_100_utm)
    focus_filter_150_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_150_utm)
    focus_filter_300_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_300_utm)
    # create xarray object of filtered dem
    focus_filter_xarray_50_utm = focus_utm.copy(data=focus_filter_50_utm)
    focus_filter_xarray_100_utm = focus_utm.copy(data=focus_filter_100_utm)
    focus_filter_xarray_150_utm = focus_utm.copy(data=focus_filter_150_utm)
    focus_filter_xarray_300_utm = focus_utm.copy(data=focus_filter_300_utm)

    # create xarray slopes
    dz_dlat_xar, dz_dlon_xar = focus_utm.differentiate(coord='y'), focus_utm.differentiate(coord='x')
    dz_dlat_filter_xar_50, dz_dlon_filter_xar_50 = focus_filter_xarray_50_utm.differentiate(coord='y'), focus_filter_xarray_50_utm.differentiate(coord='x')
    dz_dlat_filter_xar_100, dz_dlon_filter_xar_100 = focus_filter_xarray_100_utm.differentiate(coord='y'), focus_filter_xarray_100_utm.differentiate(coord='x')
    dz_dlat_filter_xar_150, dz_dlon_filter_xar_150 = focus_filter_xarray_150_utm.differentiate(coord='y'), focus_filter_xarray_150_utm.differentiate(coord='x')
    dz_dlat_filter_xar_300, dz_dlon_filter_xar_300  = focus_filter_xarray_300_utm.differentiate(coord='y'), focus_filter_xarray_300_utm.differentiate(coord='x')

    # interpolate slope and dem
    elevation_data = focus_utm.interp(y=northings_xar, x=eastings_xar, method='linear').data
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

    # Fill dataframe with elevation and slopes
    points_df['elevation_astergdem'] = elevation_data
    points_df['slope_lat'] = slope_lat_data
    points_df['slope_lon'] = slope_lon_data
    points_df['slope_lat_gf50'] = slope_lat_data_filter_50
    points_df['slope_lon_gf50'] = slope_lon_data_filter_50
    points_df['slope_lat_gf100'] = slope_lat_data_filter_100
    points_df['slope_lon_gf100'] = slope_lon_data_filter_100
    points_df['slope_lat_gf150'] = slope_lat_data_filter_150
    points_df['slope_lon_gf150'] = slope_lon_data_filter_150
    points_df['slope_lat_gf300'] = slope_lat_data_filter_300
    points_df['slope_lon_gf300'] = slope_lon_data_filter_300

    calculate_elevation_and_slopes_in_epsg_4326_and_show_differences_wrt_utm = False
    if calculate_elevation_and_slopes_in_epsg_4326_and_show_differences_wrt_utm:
        lon_c = (0.5 * (focus.coords['x'][-1] + focus.coords['x'][0])).to_numpy()
        lat_c = (0.5 * (focus.coords['y'][-1] + focus.coords['y'][0])).to_numpy()
        ris_metre_lon = haversine(lon_c, lat_c, lon_c + ris_ang, lat_c) * 1000
        ris_metre_lat = haversine(lon_c, lat_c, lon_c, lat_c + ris_ang) * 1000

        # calculate slope for restricted dem
        dz_dlat, dz_dlon = np.gradient(focus.values, -ris_metre_lat, ris_metre_lon)  # [m/m]
        dz_dlat_xarray = focus.copy(data=dz_dlat)
        dz_dlon_xarray = focus.copy(data=dz_dlon)

        # interpolate dem and slope
        elevation_data1 = focus.interp(y=lats_xar, x=lons_xar, method='linear').data  # (N,)
        slope_lat_data1 = dz_dlat_xarray.interp(y=lats_xar, x=lons_xar, method='linear').data  # (N,)
        slope_lon_data1 = dz_dlon_xarray.interp(y=lats_xar, x=lons_xar, method='linear').data  # (N,)

        assert slope_lat_data1.shape == slope_lon_data1.shape == elevation_data1.shape, "Different shapes, something wrong!"

        fig, axes = plt.subplots(2,3, figsize=(10,8))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        # elevation
        im1 = focus.plot(ax=ax1, cmap='viridis', vmin=np.nanmin(elevation_data1),
                                  vmax=np.nanmax(elevation_data1), zorder=0)
        s1 = ax1.scatter(x=lons_xar, y=lats_xar, s=50, c=elevation_data1, ec=None, cmap='viridis',
                         vmin=np.nanmin(elevation_data1), vmax=np.nanmax(elevation_data1), zorder=1)
        # slope_lat
        im2 = dz_dlat_xarray.plot(ax=ax2, cmap='viridis', vmin=np.nanmin(slope_lat_data1),
                                  vmax=np.nanmax(slope_lat_data1), zorder=0)
        s2 = ax2.scatter(x=lons_xar, y=lats_xar, s=50, c=slope_lat_data1, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lat_data1), vmax=np.nanmax(slope_lat_data1), zorder=1)
        # slope_lon
        im3 = dz_dlon_xarray.plot(ax=ax3, cmap='viridis', vmin=np.nanmin(slope_lon_data1),
                                  vmax=np.nanmax(slope_lon_data1), zorder=0)
        s3 = ax3.scatter(x=lons_xar, y=lats_xar, s=50, c=slope_lon_data1, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lon_data1), vmax=np.nanmax(slope_lon_data1), zorder=1)
        # utm elevation
        im4 = focus_utm.plot(ax=ax4, cmap='viridis', vmin=np.nanmin(elevation_data),
                                  vmax=np.nanmax(elevation_data), zorder=0)
        s4 = ax4.scatter(x=eastings_xar, y=northings_xar, s=50, c=elevation_data, ec=None, cmap='viridis',
                         vmin=np.nanmin(elevation_data), vmax=np.nanmax(elevation_data), zorder=1)
        # utm slope_lat
        im5 = dz_dlat_xar.plot(ax=ax5, cmap='viridis', vmin=np.nanmin(slope_lat_data),
                                  vmax=np.nanmax(slope_lat_data), zorder=0)
        s5 = ax5.scatter(x=eastings_xar, y=northings_xar, s=50, c=slope_lat_data, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lat_data), vmax=np.nanmax(slope_lat_data), zorder=1)
        # utm slope_lon
        im6 = dz_dlon_xar.plot(ax=ax6, cmap='viridis', vmin=np.nanmin(slope_lon_data),
                                  vmax=np.nanmax(slope_lon_data), zorder=0)
        s6 = ax6.scatter(x=eastings_xar, y=northings_xar, s=50, c=slope_lon_data, ec=None, cmap='viridis',
                         vmin=np.nanmin(slope_lon_data), vmax=np.nanmax(slope_lon_data), zorder=1)

        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.scatter(x=elevation_data1, y=elevation_data)
        ax2.scatter(x=slope_lon_data1, y=slope_lon_data)
        ax3.scatter(x=slope_lat_data1, y=slope_lat_data)
        l1 = ax1.plot([0, 2000], [0, 2000], color='red', linestyle='--')
        l2 = ax2.plot([-3, 3], [-3, 3], color='red', linestyle='--')
        l3 = ax3.plot([-2, 2], [-2, 2], color='red', linestyle='--')
        plt.show()


    """ Calculate Farinotti ith_f """
    print(f"Calculating ith_f...")
    # Set farinotti ice thickness folder
    folder_rgi_farinotti = args.farinotti_icethickness_folder + f'composite_thickness_RGI60-{rgi:02d}/RGI60-{rgi:02d}/'
    try: # Import farinotti ice thickness file. Note that it contains zero where ice not present.
        file_glacier_farinotti =rioxarray.open_rasterio(f'{folder_rgi_farinotti}{glacier_name}_thickness.tif', masked=False)
        file_glacier_farinotti = file_glacier_farinotti.where(file_glacier_farinotti != 0.0) # replace zeros with nans.
        file_glacier_farinotti.rio.write_nodata(np.nan, inplace=True)
    except:
        print(f"No Farinotti data can be found for rgi {rgi} glacier {glacier_name}")
        input('check.')

    transformer = Transformer.from_crs("EPSG:4326", file_glacier_farinotti.rio.crs)
    lons_crs_f, lats_crs_f = transformer.transform(points_df['lats'].to_numpy(), points_df['lons'].to_numpy())

    try:
        ith_f_data = file_glacier_farinotti.interp(y=xarray.DataArray(lats_crs_f), x=xarray.DataArray(lons_crs_f),
                                    method="nearest").data.squeeze()
        points_df['ith_f'] = ith_f_data
        print(f"From Farinotti ith interpolation we have generated {np.isnan(ith_f_data).sum()} nans.")
        no_farinotti_data = False
    except:
        print(f"Farinotti interpolation rgi {rgi} glacier {glacier_name} is problematic. Check")
        no_farinotti_data = True


    """ Calculate Millan vx, vy, v """
    print(f"Calculating vx, vy, v, ith_m...")

    # get Millan vx files for specific rgi and create vx mosaic
    files_vx = glob(args.millan_velocity_folder + 'RGI-{}/VX_RGI-{}*'.format(rgi, rgi))
    mosaic_vx_dict = {'list_vx': [], 'epsg_utm_vx': []}
    for tiffile in files_vx:
        xds = rioxarray.open_rasterio(tiffile, masked=False)
        xds.rio.write_nodata(np.nan, inplace=True)
        mosaic_vx_dict['list_vx'].append(xds)
        mosaic_vx_dict['epsg_utm_vx'].append(xds.rio.crs)

    # check
    if len(mosaic_vx_dict['epsg_utm_vx']) > 1:
        all_same_epsg = all(x == mosaic_vx_dict['epsg_utm_vx'][0] for x in mosaic_vx_dict['epsg_utm_vx'])
        if not all_same_epsg:
            raise ValueError("In Millan you are trying to mosaic files with different epsgs.")

    # get Millan vy files for specific rgi and create vy mosaic
    files_vy = glob(args.millan_velocity_folder + 'RGI-{}/VY_RGI-{}*'.format(rgi, rgi))
    mosaic_vy_dict = {'list_vy': [], 'epsg_utm_vy': []}
    for tiffile in files_vy:
        xds = rioxarray.open_rasterio(tiffile, masked=False)
        xds.rio.write_nodata(np.nan, inplace=True)
        mosaic_vy_dict['list_vy'].append(xds)

    # get Millan ice thickness files for specific rgi and create ith mosaic
    files_ith = glob(args.millan_icethickness_folder + 'RGI-{}/THICKNESS_RGI-{}*'.format(rgi, rgi))
    mosaic_ith_dict = {'list_ith': [], 'epsg_utm_ith': []}
    for tiffile in files_ith:
        xds = rioxarray.open_rasterio(tiffile, masked=False)
        xds.rio.write_nodata(np.nan, inplace=True)
        mosaic_ith_dict['list_ith'].append(xds)

    mosaic_vx = merge.merge_arrays(mosaic_vx_dict['list_vx'])
    mosaic_vy = merge.merge_arrays(mosaic_vy_dict['list_vy'])
    mosaic_ith = merge.merge_arrays(mosaic_ith_dict['list_ith'])

    # check
    assert mosaic_vx.rio.crs == mosaic_vy.rio.crs == mosaic_ith.rio.crs, 'Millan - sth wrong in espg of mosaics'

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

    ris_metre_millan = mosaic_vx.rio.resolution()[0]
    crs = mosaic_vx.rio.crs
    eps_millan = 10 * ris_metre_millan
    transformer = Transformer.from_crs("EPSG:4326", crs)

    # Covert lat lon coordinates to Millan projection
    lons_crs, lats_crs = transformer.transform(points_df['lats'].to_numpy(), points_df['lons'].to_numpy())

    lons_crs = xarray.DataArray(lons_crs)
    lats_crs = xarray.DataArray(lats_crs)

    # clip millan mosaic around the generated points
    try:
        focus_vx = mosaic_vx.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                          maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan)
        focus_vy = mosaic_vy.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                          maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan)
        focus_ith = mosaic_ith.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                          maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan)

        focus_vx = focus_vx.squeeze()
        focus_vy = focus_vy.squeeze()
        focus_ith = focus_ith.squeeze()

        # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
        num_px_sigma_50 = max(1, round(50 / ris_metre_millan))  # 1
        num_px_sigma_100 = max(1, round(100 / ris_metre_millan))  # 2
        num_px_sigma_150 = max(1, round(150 / ris_metre_millan))  # 3
        num_px_sigma_300 = max(1, round(300 / ris_metre_millan))  # 6

        # Apply filter to velocities
        focus_filter_vx_50 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_50)
        focus_filter_vx_100 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_100)
        focus_filter_vx_150 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_150)
        focus_filter_vx_300 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_300)
        focus_filter_vy_50 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_50)
        focus_filter_vy_100 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_100)
        focus_filter_vy_150 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_150)
        focus_filter_vy_300 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_300)

        # Mask back the filtered arrays
        focus_filter_vx_50 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_50)
        focus_filter_vx_100 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_100)
        focus_filter_vx_150 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_150)
        focus_filter_vx_300 = np.where(np.isnan(focus_vx.values), np.nan, focus_filter_vx_300)
        focus_filter_vy_50 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_50)
        focus_filter_vy_100 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_100)
        focus_filter_vy_150 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_150)
        focus_filter_vy_300 = np.where(np.isnan(focus_vy.values), np.nan, focus_filter_vy_300)

        # create xarrays of filtered velocities
        focus_filter_vx_50_ar = focus_vx.copy(data=focus_filter_vx_50)
        focus_filter_vx_100_ar = focus_vx.copy(data=focus_filter_vx_100)
        focus_filter_vx_150_ar = focus_vx.copy(data=focus_filter_vx_150)
        focus_filter_vx_300_ar = focus_vx.copy(data=focus_filter_vx_300)
        focus_filter_vy_50_ar = focus_vy.copy(data=focus_filter_vy_50)
        focus_filter_vy_100_ar = focus_vy.copy(data=focus_filter_vy_100)
        focus_filter_vy_150_ar = focus_vy.copy(data=focus_filter_vy_150)
        focus_filter_vy_300_ar = focus_vy.copy(data=focus_filter_vy_300)

        # Interpolate (note: nans can be produced near boundaries). This should be removed at the end.
        ith_data = focus_ith.interp(y=lats_crs, x=lons_crs, method="nearest").data
        vx_data = focus_vx.interp(y=lats_crs, x=lons_crs, method="nearest").data
        vy_data = focus_vy.interp(y=lats_crs, x=lons_crs, method="nearest").data
        vx_filter_50_data = focus_filter_vx_50_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vx_filter_100_data = focus_filter_vx_100_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vx_filter_150_data = focus_filter_vx_150_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vx_filter_300_data = focus_filter_vx_300_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vy_filter_50_data = focus_filter_vy_50_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vy_filter_100_data = focus_filter_vy_100_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vy_filter_150_data = focus_filter_vy_150_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vy_filter_300_data = focus_filter_vy_300_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data


        print(
            f"From Millan vx, vy, ith interpolations we have generated {np.isnan(vx_data).sum()}/{np.isnan(vy_data).sum()}/{np.isnan(ith_data).sum()} nans.")

        # Fill dataframe with vx, vy, ith_m etc
        # Note this vectors may contain nans from interpolation at the margin/inside nunatak
        points_df['ith_m'] = ith_data
        points_df['vx'] = vx_data
        points_df['vy'] = vy_data
        points_df['vx_gf50'] = vx_filter_50_data
        points_df['vx_gf100'] = vx_filter_100_data
        points_df['vx_gf150'] = vx_filter_150_data
        points_df['vx_gf300'] = vx_filter_300_data
        points_df['vy_gf50'] = vy_filter_50_data
        points_df['vy_gf100'] = vy_filter_100_data
        points_df['vy_gf150'] = vy_filter_150_data
        points_df['vy_gf300'] = vy_filter_300_data
        no_millan_data = False
    except:
        print(f"No Millan data can be found for rgi {rgi} glacier {glacier_name}")
        no_millan_data = True
        # Data imputation: set Millan velocities as zero (keep ith_m as nan)
        for col in ['vx','vy','vx_gf50', 'vx_gf100', 'vx_gf150', 'vx_gf300', 'vy_gf50', 'vy_gf100', 'vy_gf150', 'vy_gf300']:
            points_df[col] = 0.0

    """ Calculate distance_from_border """
    print(f"Calculating the distances using glacier geometries... ")

    # Get the UTM EPSG code from glacier center coordinates
    cenLon, cenLat = gl_df['CenLon'].item(), gl_df['CenLat'].item()
    _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(cenLat, cenLon)

    # Create Geopandas geoseries objects of glacier geometries (boundary and nunataks) and convert to UTM
    if list_cluster_RGIIds is None: # Case 1: isolated glacier
        print(f"Isolated glacier")
        exterior_ring = gl_geom.exterior  # shapely.geometry.polygon.LinearRing
        interior_rings = gl_geom.interiors  # shapely.geometry.polygon.InteriorRingSequence of polygon.LinearRing
        geoseries_geometries_4326 = gpd.GeoSeries([exterior_ring] + list(interior_rings), crs="EPSG:4326")
        geoseries_geometries_epsg = geoseries_geometries_4326.to_crs(epsg=glacier_epsg)

    elif list_cluster_RGIIds is not None: # Case 2: cluster of glaciers with ice divides
        print(f"Cluster of glaciers with ice divides.")
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
            cluster_interior_rings = list(cluster_geometry_no_divides_epsg.item().interiors)  # shapely.geometry.polygon.LinearRing
            multipolygon = False
        elif cluster_geometry_no_divides_epsg.item().geom_type == 'MultiPolygon':
            polygons = list(cluster_geometry_no_divides_epsg.item().geoms)
            cluster_exterior_ring = [polygon.exterior for polygon in polygons]  # list of shapely.geometry.polygon.LinearRing
            num_multipoly = len(cluster_exterior_ring)
            cluster_interior_ringSequences = [polygon.interiors for polygon in polygons]  # list of shapely.geometry.polygon.InteriorRingSequence
            cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence]  # list of shapely.geometry.polygon.LinearRing
            multipolygon = True
        else: raise ValueError("Unexpected geometry type. Please check.")

        # Create a geoseries of all external and internal geometries
        geoseries_geometries_epsg = gpd.GeoSeries(cluster_exterior_ring + cluster_interior_rings)


    # Get all generated points and create Geopandas geoseries and convert to UTM
    list_points = [Point(lon, lat) for (lon, lat) in zip(points_df['lons'], points_df['lats'])]
    geoseries_points_4326 = gpd.GeoSeries(list_points, crs="EPSG:4326")
    geoseries_points_epsg = geoseries_points_4326.to_crs(epsg=glacier_epsg)

    # Loop over generated points
    for (i, lon, lat, nunatak) in zip(points_df.index, points_df['lons'], points_df['lats'], points_df['nunataks']):

        # Make a check.
        easting, nothing, zonenum, zonelett, epsg = from_lat_lon_to_utm_and_epsg(lat, lon)
        if epsg != glacier_epsg:
            print(f"Note differet UTM zones. Point espg {epsg} and glacier center epsg {glacier_epsg}.")

        # Get shapely Point
        point_epsg = geoseries_points_epsg.iloc[i]

        # Calculate the distances between such point and all glacier geometries
        min_distances_point_geometries = geoseries_geometries_epsg.distance(point_epsg)
        min_dist = np.min(min_distances_point_geometries) # unit UTM: m

        # To debug we want to check what point corresponds to the minimum distance.
        debug_distance = True
        if debug_distance:
            min_distance_index = min_distances_point_geometries.idxmin()
            nearest_line = geoseries_geometries_epsg.loc[min_distance_index]
            nearest_point_on_line = nearest_line.interpolate(nearest_line.project(point_epsg))
            # print(f"{i} Minimum distance: {min_dist:.2f} meters.")

        # Fill dataset
        # note that the generated points cannot be in nunataks so distances are well defined
        points_df.loc[i, 'dist_from_border_km_geom'] = min_dist/1000.

        # Plot
        plot_calculate_distance = False
        if plot_calculate_distance:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.plot(*gl_geom_ext.exterior.xy, lw=1, c='red')
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
                    ax1.plot(*intersect.xy, lw=1, color='k') #np.random.rand(3)

                # Plot cluster ice divides removed
                if multipolygon:
                    polygons = list(cluster_geometry_no_divides_4326.item().geoms)
                    cluster_exterior_ring = [polygon.exterior for polygon in polygons]  # list of shapely.geometry.polygon.LinearRing
                    cluster_interior_ringSequences = [polygon.interiors for polygon in polygons]  # list of shapely.geometry.polygon.InteriorRingSequence
                    cluster_interior_rings = [ring for sequence in cluster_interior_ringSequences for ring in sequence]  # list of shapely.geometry.polygon.LinearRing
                    for exterior in cluster_exterior_ring:
                        ax1.plot(*exterior.xy, lw=1, c='red', zorder=3)
                    for interior in cluster_interior_rings:
                        ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

                else:
                    ax1.plot(*cluster_geometry_no_divides_4326.item().exterior.xy, lw=1, c='red', zorder=3)
                    for interior in cluster_geometry_no_divides_4326.item().interiors:
                        ax1.plot(*interior.xy, lw=1, c='blue', zorder=3)

            if nunatak: ax1.scatter(lon, lat, s=50, lw=2, c='b')
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


            if nunatak: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='b')
            else: ax2.scatter(*point_epsg.xy, s=50, lw=2, c='r', ec='r')
            if debug_distance: ax2.scatter(*nearest_point_on_line.xy, s=50, lw=2, c='g')

            ax1.set_title('EPSG 4326')
            ax2.set_title(f'EPSG {glacier_epsg}')
            plt.show()

    # convert this column to float (misteriously it was an object type)
    points_df['dist_from_border_km_geom'] = pd.to_numeric(points_df['dist_from_border_km_geom'], errors='coerce')

    print(f"Finished distance calculations.")

    # Show the result
    show_glacier_with_produced_points = False
    if show_glacier_with_produced_points:
        fig, axes = plt.subplots(2,3, figsize=(10,8))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        for ax in (ax1, ax2):
            ax.plot(*gl_geom_ext.exterior.xy, lw=1, c='red')
            for interior in gl_geom.interiors:
                ax.plot(*interior.xy, lw=1, c='blue')
            for (lon, lat, nunatak) in zip(points['lons'], points['lats'], points['nunataks']):
                if nunatak: ax.scatter(lon, lat, s=50, lw=2, c='magenta', zorder=2)
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
        if no_millan_data is False:
            im3 = focus_vx.plot(ax=ax3, cmap='viridis', vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data))
            s3 = ax3.scatter(x=lons_crs, y=lats_crs, s=50, c=vx_data, ec=(1, 0, 0, 1), cmap='viridis',
                             vmin=np.nanmin(vx_data), vmax=np.nanmax(vx_data), zorder=1)
            s3_1 = ax3.scatter(x=lons_crs[np.argwhere(np.isnan(vx_data))], y=lats_crs[np.argwhere(np.isnan(vx_data))], s=50,
                               c='magenta', zorder=1)

        # farinotti
        if no_farinotti_data is False:
            im4 = file_glacier_farinotti.plot(ax=ax4, cmap='inferno', vmin=np.nanmin(file_glacier_farinotti),
                                              vmax=np.nanmax(file_glacier_farinotti))
            s4 = ax4.scatter(x=lons_crs_f, y=lats_crs_f, s=50, c=ith_f_data, ec=(1, 0, 0, 1), cmap='inferno',
                             vmin=np.nanmin(file_glacier_farinotti), vmax=np.nanmax(file_glacier_farinotti), zorder=1)
            s4_1 = ax4.scatter(x=lons_crs_f[np.argwhere(np.isnan(ith_f_data))],
                               y=lats_crs_f[np.argwhere(np.isnan(ith_f_data))], s=50, c='magenta', zorder=2)

        # distance
        if multipolygon:
            for i_poly in range(num_multipoly):
                ax5.plot(*geoseries_geometries_epsg.loc[i_poly].xy, lw=1, c='red')  # first num_multipoly are outside borders
            for inter in geoseries_geometries_epsg.loc[num_multipoly:]:  # all interiors if present
                ax5.plot(*inter.xy, lw=1, c='blue')
        else:
            ax5.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='red')  # first entry is outside border
            for inter in geoseries_geometries_epsg.loc[1:]:  # all interiors if present
                ax5.plot(*inter.xy, lw=1, c='blue')

        ax5.scatter(x=geoseries_points_epsg.x, y=geoseries_points_epsg.y, s=5, lw=2, cmap='cividis',
                    c=points_df['dist_from_border_km_geom'],  vmin=points_df['dist_from_border_km_geom'].min(),
                    vmax=points_df['dist_from_border_km_geom'].max())

        ax6.axis('off')

        for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            ax.axis("off")

        plt.tight_layout()
        plt.show()


    """ Cleaning the produced dataset """
    # At this stage any nan may be present in vx, vy, v, ith_m, ith_f. Remove those points.
    points_df = points_df.dropna(subset=['vx', 'vy', 'ith_m', 'ith_f'])
    #print(points_df.T)
    print(f"Generated dataset. Nan present: {points_df.isnull().any().any()}")
    print(f"*******FINISHED FETCHING FEATURES*******")
    return points_df


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

# generated_points_dataframe = populate_glacier_with_metadata(glacier_name='RGI60-07.00832', n=2000)

# 'RGI60-07.00228' should be a multiplygon
# RGI60-11.00781 has only 1 neighbor
# RGI60-08.00001 has no Millan data
# RGI60-11.00846 has multiple intersects with neighbors
# RGI60-11.02774 has no neighbors
#RGI60-11.02884 has no neighbors
#'RGI60-11.01450' Aletsch # RGI60-11.02774
#RGI60-11.00590, RGI60-11.01894 no Millan data ?
#glacier_name = np.random.choice(RGI_burned)