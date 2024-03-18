import os, sys
from glob import glob
import argparse
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import xarray, rioxarray, rasterio
import xrspatial.curvature
import xrspatial.aspect
from rioxarray import merge
import geopandas as gpd
import oggm
from oggm import utils
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from pyproj import Proj, Transformer
from math import radians, cos, sin, asin, sqrt, floor
import utm
import time
from rtree import index
from joblib import Parallel, delayed

from create_rgi_mosaic_tanxedem import fetch_dem, find_tandemx_tiles
from utils_metadata import haversine, from_lat_lon_to_utm_and_epsg, gaussian_filter_with_nans

"""
This program generates glacier metadata at some random locations inside the glacier geometry. 

Input: glacier name (RGIId), how many points you want to generate. 
Output: pandas dataframe with features calculated for each generated point. 
Computational time scales linearly with no. generated points- Ca. 50% time is point generation, other 50% distances. 

Note: the features slope, elevation_from_zmin and v are calculated in model.py, not here.

Note: the points are generated inside the glacier but outside nunataks (there is a check for this)

Note: as of Feb 16, 2024 I decide to fill the nans in Millan veloity fields. I do that interpolating these fields.
After that I interpolate at the locations of the generated points.

Note: Millan and Farinotti products needs to be interpolated. Interpolation close to the borders may result in nans. 
The interpolation method="nearest" yields much less nans close to borders if compared to linear
interpolation and therefore is chosen. 

Note that Farinotti interpolation ith_f may result in nan when generated point too close to the border.

Note the following policy for Millan special cases to produce vx, vy, v, ith_m:
    1) There is no Millan data for such glacier. Data imputation: vy=vy=v=0.0 and ith_m=nan. 
    2) In case the interpolation of Millan's fields yields nan because points are either too close to the margins. 
    I keep the nans that will be however removed before returning the dataset.   
"""
# todo: Data imputation: Millan and other features
# todo: inserire anche un ulteriore feature che è la velocità media di tutto il ghiacciao ? sia vxm, vym, vm ?

# todo: a proposito di come smussare i campi di slope e velocita, guardare questo articolo:
#  Slope estimation influences on ice thickness inversion models: a case study for Monte Tronador glaciers, North Patagonian Andes

# todo: 1. implement faster version of slope calculation
# todo: 2. implement faster version of velocity calculation

parser = argparse.ArgumentParser()
parser.add_argument('--mosaic', type=str,default="/media/nico/samsung_nvme/Tandem-X-EDEM/",
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


def populate_glacier_with_metadata(glacier_name, dem_rgi=None, n=50, seed=None):
    print(f"******* FETCHING FEATURES FOR GLACIER {glacier_name} *******")
    tin=time.time()

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

    try:
        # Get glacier dataset
        gl_df = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name]
        # Get the UTM EPSG code from glacier center coordinates
        cenLon, cenLat = gl_df['CenLon'].item(), gl_df['CenLat'].item()
        _, _, _, _, glacier_epsg = from_lat_lon_to_utm_and_epsg(cenLat, cenLon)
        print(f"Glacier {glacier_name} found. Lat: {cenLat}, Lon: {cenLon}")
        assert len(gl_df) == 1, "Check this please."
        # print(gl_df.T)
    except Exception as e:
        print(f"Error. {glacier_name} not present in OGGM's RGI v6.")
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

    # Geodataframes of external boundary and all internal nunataks
    gl_geom_nunataks_gdf = gpd.GeoDataFrame(geometry=gl_geom_nunataks_list, crs="EPSG:4326")
    gl_geom_ext_gdf = gpd.GeoDataFrame(geometry=[gl_geom_ext], crs="EPSG:4326")

    tgeometries = time.time() - tin
    print(f"Generating {n} points...")
    tp0 = time.time()
    # Generate points (no points can be generated inside nunataks)
    points = {'lons': [], 'lats': [], 'nunataks': []}
    if seed is not None: np.random.seed(seed)

    while (len(points['lons']) < n):
        batch_size = min(n, n - len(points['lons']))  # Adjust batch size as needed
        r_lons = np.random.uniform(llx, urx, batch_size)
        r_lats = np.random.uniform(lly, ury, batch_size)
        points_batch = list(map(Point, r_lons, r_lats))
        points_batch_gdf = gpd.GeoDataFrame(geometry=points_batch, crs="EPSG:4326")

        # 1) First we select only those points generated inside the glacier
        points_yes_no_ext_gdf = gpd.sjoin(points_batch_gdf, gl_geom_ext_gdf, how="left", op="within")
        points_in_glacier_gdf = points_yes_no_ext_gdf[~points_yes_no_ext_gdf.index_right.isna()].drop(columns=['index_right'])

        indexes_of_points_inside = points_in_glacier_gdf.index

        # 2) Then we get rid of all those generated inside nunataks
        points_yes_no_nunataks_gdf = gpd.sjoin(points_batch_gdf.loc[indexes_of_points_inside], gl_geom_nunataks_gdf, how="left", op="within")
        points_not_in_nunataks_gdf = points_yes_no_nunataks_gdf[points_yes_no_nunataks_gdf.index_right.isna()].drop(columns=['index_right'])

        points['lons'].extend(points_not_in_nunataks_gdf['geometry'].x.tolist())
        points['lats'].extend(points_not_in_nunataks_gdf['geometry'].y.tolist())
        points['nunataks'].extend([0.0]*len(points_not_in_nunataks_gdf))


        plot_gen_points = False
        if plot_gen_points:
            points_in_nunataks_gdf = points_yes_no_nunataks_gdf[~points_yes_no_nunataks_gdf.index_right.isna()].drop(
                columns=['index_right'])
            fig, ax = plt.subplots()
            ax.plot(*gl_geom.exterior.xy, color='blue')
            gl_geom_nunataks_gdf.plot(ax=ax, color='orange', alpha=0.5)
            #points_in_glacier_gdf.plot(ax=ax, color='red', alpha=0.5, markersize=1, zorder=2)
            points_not_in_nunataks_gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=1, zorder=2)
            points_in_nunataks_gdf.plot(ax=ax, color='red', alpha=0.5, markersize=1, zorder=2)
            plt.show()

    print(f"We have generated {len(points['lats'])} points.")
    # Feature dataframe
    points_df = pd.DataFrame(columns=['lons', 'lats', 'nunataks'])
    # Fill lats, lons and nunataks
    points_df['lats'] = points['lats']
    points_df['lons'] = points['lons']
    points_df['nunataks'] = points['nunataks']
    if (points_df['nunataks'].sum() != 0):
        print(f"The generation pipeline has produced n. {points_df['nunataks'].sum()} points inside nunataks")
        raise ValueError

    tp1 = time.time()
    tgenpoints = tp1-tp0

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

    """ Calculate Millan vx, vy, v """
    print(f"Calculating vx, vy, v, ith_m...")
    tmillan1 = time.time()

    # get Millan files
    files_vx = sorted(glob(f"{args.millan_velocity_folder}RGI-{rgi}/VX_RGI-{rgi}*"))
    files_vy = sorted(glob(f"{args.millan_velocity_folder}RGI-{rgi}/VY_RGI-{rgi}*"))
    files_ith = sorted(glob(f"{args.millan_icethickness_folder}RGI-{rgi}/THICKNESS_RGI-{rgi}*"))


    mosaic_vx, mosaic_vy, mosaic_ith = None, None, None

    for i, (file_vx, file_vy, file_ith) in enumerate(zip(files_vx, files_vy, files_ith)):

        tile_vx = rioxarray.open_rasterio(file_vx, masked=False)
        tile_vy = rioxarray.open_rasterio(file_vy, masked=False)
        tile_ith = rioxarray.open_rasterio(file_ith, masked=False)

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

            #print(f'{i} Vx and ith bounds are different! Re-aligning them.')
            #print(tile_vx.rio.bounds(), tile_vx.shape)
            #print(tile_ith.rio.bounds(), tile_ith.shape)

            #tile_vx_aligned, tile_ith_aligned = xarray.align(tile_vx, tile_ith, join="outer")
            tile_ith = tile_ith.reindex_like(tile_vx, method="nearest", tolerance=50., fill_value=np.nan)
            #print(tile_ith.rio.bounds(), tile_ith.shape)

            #fig, (ax1, ax2) = plt.subplots(1,2)
            #tile_vx.plot(ax=ax1, cmap='viridis')
            #tile_ith.plot(ax=ax2, cmap='viridis')
            #plt.show()

        assert tile_vx.rio.bounds() == tile_vy.rio.bounds() == tile_vx.rio.bounds(), "All tiles bounds not the same"

        # Convert tile boundaries to lat lon
        tran = Transformer.from_crs(tile_vx.rio.crs, "EPSG:4326")
        lat0, lon0 = tran.transform(tile_vx.rio.bounds()[0], tile_vx.rio.bounds()[1])
        lat1, lon1 = tran.transform(tile_vx.rio.bounds()[2], tile_vx.rio.bounds()[3])
        #print(f"Glacier bounds: {llx, lly, urx, ury}, tile bounds: {lon0, lat0, lon1, lat1}")

        # Check if tile contains glacier. If yes, we have found our guy (no point in investigating the other tiles)
        is_glacier_inside_tile = lon0 < llx < urx < lon1 and lat0 < lly < ury < lat1
        if is_glacier_inside_tile:
            #print(f'Tile {i}: glacier included: {is_glacier_inside_tile}')
            mosaic_vx = tile_vx
            mosaic_vy = tile_vy
            mosaic_ith = tile_ith
            break

    if mosaic_vx is None and mosaic_vy is None and mosaic_ith is None:
        no_millan_data = True

    else:
        # Mask ice thickness based on velocity map
        mosaic_ith = mosaic_ith.where(mosaic_vx.notnull())

        # Covert lat lon coordinates of the points to Millan's projection
        ris_metre_millan = mosaic_vx.rio.resolution()[0]
        eps_millan = 10 * ris_metre_millan
        #print(f"Ris metre millan: {ris_metre_millan}, eps: {eps_millan}")
        crs = mosaic_vx.rio.crs

        transformer = Transformer.from_crs("EPSG:4326", crs)
        lons_crs, lats_crs = transformer.transform(points_df['lats'].to_numpy(), points_df['lons'].to_numpy())

        plot_millan_glacier_data = False
        if plot_millan_glacier_data:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            mosaic_vx.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                   maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan).plot(ax=ax1,
                                                                                                                cmap='viridis')
            mosaic_vy.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                   maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan).plot(ax=ax2,
                                                                                                                cmap='viridis')
            mosaic_ith.rio.clip_box(minx=np.min(lons_crs) - eps_millan, miny=np.min(lats_crs) - eps_millan,
                                    maxx=np.max(lons_crs) + eps_millan, maxy=np.max(lats_crs) + eps_millan).plot(ax=ax3,
                                                                                                                 cmap='viridis')
            plt.show()

        # clip millan mosaic around the generated points
        try:
            focus_vx0 = mosaic_vx.rio.clip_box(minx=np.min(lons_crs) - eps_millan,
                                               miny=np.min(lats_crs) - eps_millan,
                                               maxx=np.max(lons_crs) + eps_millan,
                                               maxy=np.max(lats_crs) + eps_millan)
            focus_vy0 = mosaic_vy.rio.clip_box(minx=np.min(lons_crs) - eps_millan,
                                               miny=np.min(lats_crs) - eps_millan,
                                               maxx=np.max(lons_crs) + eps_millan,
                                               maxy=np.max(lats_crs) + eps_millan)
            focus_ith = mosaic_ith.rio.clip_box(minx=np.min(lons_crs) - eps_millan,
                                                miny=np.min(lats_crs) - eps_millan,
                                                maxx=np.max(lons_crs) + eps_millan,
                                                maxy=np.max(lats_crs) + eps_millan)

            no_millan_data = focus_vx0.isnull().all().item()  # If vx is empty we have no Millan data

        except:
            print("ATTENTION: no_millan_data = True")
            no_millan_data = True


    if no_millan_data is False:

        # Crucial here. I interpolate vx, vy to remove nans (as much as possible).
        # This is because velocity fields often have holes. On the other hand we keep the nans in the ice thickness ith
        focus_vx = focus_vx0.rio.interpolate_na(method='linear').squeeze()
        focus_vy = focus_vy0.rio.interpolate_na(method='linear').squeeze()
        focus_ith = focus_ith.squeeze()

        plot_Millan_interpolated = False
        if plot_Millan_interpolated:
            fig, axes = plt.subplots(1, 3)
            ax1, ax2, ax3 = axes.flatten()
            im1 = focus_vx0.plot(ax=ax1, cmap='viridis')
            im2 = focus_vx.plot(ax=ax2, cmap='viridis')
            im3 = focus_ith.plot(ax=ax3, cmap='viridis')
            plt.show()

        # Calculate how many pixels I need for a resolution of 50, 100, 150, 300 meters
        num_px_sigma_50 = max(1, round(50 / ris_metre_millan))  # 1
        num_px_sigma_100 = max(1, round(100 / ris_metre_millan))  # 2
        num_px_sigma_150 = max(1, round(150 / ris_metre_millan))  # 3
        num_px_sigma_300 = max(1, round(300 / ris_metre_millan))  # 6

        # Apply filter to velocities
        focus_filter_vx_50 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_50, trunc=4.0)
        focus_filter_vx_100 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_100, trunc=4.0)
        focus_filter_vx_150 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_150, trunc=4.0)
        focus_filter_vx_300 = gaussian_filter_with_nans(U=focus_vx.values, sigma=num_px_sigma_300, trunc=4.0)
        focus_filter_vy_50 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_50, trunc=4.0)
        focus_filter_vy_100 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_100, trunc=4.0)
        focus_filter_vy_150 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_150, trunc=4.0)
        focus_filter_vy_300 = gaussian_filter_with_nans(U=focus_vy.values, sigma=num_px_sigma_300, trunc=4.0)

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
        focus_filter_vx_50_ar = focus_vx.copy(deep=True, data=focus_filter_vx_50)
        focus_filter_vx_100_ar = focus_vx.copy(deep=True,data=focus_filter_vx_100)
        focus_filter_vx_150_ar = focus_vx.copy(deep=True,data=focus_filter_vx_150)
        focus_filter_vx_300_ar = focus_vx.copy(deep=True,data=focus_filter_vx_300)
        focus_filter_vy_50_ar = focus_vy.copy(deep=True,data=focus_filter_vy_50)
        focus_filter_vy_100_ar = focus_vy.copy(deep=True,data=focus_filter_vy_100)
        focus_filter_vy_150_ar = focus_vy.copy(deep=True,data=focus_filter_vy_150)
        focus_filter_vy_300_ar = focus_vy.copy(deep=True,data=focus_filter_vy_300)

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
        vy_filter_50_data = focus_filter_vy_50_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vy_filter_100_data = focus_filter_vy_100_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vy_filter_150_data = focus_filter_vy_150_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        vy_filter_300_data = focus_filter_vy_300_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data

        dvx_dx_data = dvx_dx_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        dvx_dy_data = dvx_dy_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        dvy_dx_data = dvy_dx_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data
        dvy_dy_data = dvy_dy_ar.interp(y=lats_crs, x=lons_crs, method='nearest').data

        print(f"From Millan vx, vy, ith interpolations we have generated "
              f"{np.isnan(vx_data).sum()}/{np.isnan(vy_data).sum()}/{np.isnan(ith_data).sum()}/"
              f"{np.isnan(vx_filter_300_data).sum()}/{np.isnan(vy_filter_300_data).sum()}/"
              f"{np.isnan(dvx_dx_data).sum()}/{np.isnan(dvx_dy_data).sum()}/"
              f"{np.isnan(dvy_dx_data).sum()}/{np.isnan(dvy_dy_data).sum()} nans.")


        # Fill dataframe with vx, vy, ith_m etc
        # Note this vectors may contain nans from interpolation close to margin/nunatak
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
        points_df['dvx_dx'] = dvx_dx_data
        points_df['dvx_dy'] = dvx_dy_data
        points_df['dvy_dx'] = dvy_dx_data
        points_df['dvy_dy'] = dvy_dy_data

    elif no_millan_data is True:
        print(f"No Millan data can be found for rgi {rgi} glacier {glacier_name}. Data imputation needed !")
        points_df['ith_m'] = np.nan
        # Data imputation: set Millan velocities as zero (keep ith_m as nan)
        for col in ['vx','vy','vx_gf50', 'vx_gf100', 'vx_gf150', 'vx_gf300', 'vy_gf50', 'vy_gf100', 'vy_gf150', 'vy_gf300',
                    'dvx_dx', 'dvx_dy', 'dvy_dx', 'dvy_dy']:
            points_df[col] = 0.0

    ifplot_millan = False
    if ifplot_millan:
        fig, axes = plt.subplots(1, 6, figsize=(10,4))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        im1 = focus_vx0.plot(ax=ax1, cmap='viridis')

        im2 = focus_vx.plot(ax=ax2, cmap='viridis')

        im3 = focus_filter_vx_50_ar.plot(ax=ax3, cmap='viridis')
        im4 = focus_filter_vy_300_ar.plot(ax=ax4, cmap='viridis')
        im5 = dvx_dx_ar.plot(ax=ax5, cmap='viridis')
        im6 = focus_ith.plot(ax=ax6, cmap='viridis')

        for ax in axes.flatten():
            ax.scatter(x=lons_crs, y=lats_crs, s=20, c='k', alpha=.1, zorder=1)

        plt.tight_layout()
        plt.show()

    tmillan2 = time.time()
    tmillan = tmillan2-tmillan1

    """ Add Slopes and Elevation """
    print(f"Calculating slopes and elevations...")
    tslope1 = time.time()
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

    # WORK IN PROGRESS
    #sth = find_tandemx_tiles(minx=swlon - (deltalon + eps),
    #                        miny=swlat - (deltalat + eps),
    #                        maxx=nelon + (deltalon + eps),
    #                        maxy=nelat + (deltalat + eps),
    #                         rgi=rgi, path_tandemx=args.mosaic)

    #input('wait')

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
    focus_utm = focus.rio.reproject(glacier_epsg, resampling=rasterio.enums.Resampling.bilinear, nodata=-9999)
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

    min_sigma, max_sigma = 100.0, 2000.0
    try:
        area_gl = points_df['Area'][0]
        lmax_gl = points_df['Lmax'][0]
        a = 1e6 * area_gl / (np.pi * 0.5 * lmax_gl)
        value = int(min(max(a, min_sigma), max_sigma))
        # print(area_gl, lmax_gl, a, value)
    except Exception as e:
        value = min_sigma
    # Ensure that our value correctly in range
    assert min_sigma <= value <= max_sigma, f"Value {value} is not within the range [{min_sigma}, {max_sigma}]"

    num_px_sigma_50 = max(1, round(50 / res_utm_metres))
    num_px_sigma_100 = max(1, round(100 / res_utm_metres))
    num_px_sigma_150 = max(1, round(150 / res_utm_metres))
    num_px_sigma_300 = max(1, round(300 / res_utm_metres))
    num_px_sigma_450 = max(1, round(450 / res_utm_metres))
    num_px_sigma_af = max(1, round(value / res_utm_metres))

    # Apply filter (utm here)
    focus_filter_50_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_50, trunc=4.0)
    focus_filter_100_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_100, trunc=4.0)
    focus_filter_150_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_150, trunc=4.0)
    focus_filter_300_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_300, trunc=4.0)
    focus_filter_450_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_450, trunc=4.0)
    focus_filter_af_utm = gaussian_filter_with_nans(U=focus_utm.values, sigma=num_px_sigma_af, trunc=3.0)

    # Mask back the filtered arrays
    focus_filter_50_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_50_utm)
    focus_filter_100_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_100_utm)
    focus_filter_150_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_150_utm)
    focus_filter_300_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_300_utm)
    focus_filter_450_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_450_utm)
    focus_filter_af_utm = np.where(np.isnan(focus_utm.values), np.nan, focus_filter_af_utm)
    # create xarray object of filtered dem
    focus_filter_xarray_50_utm = focus_utm.copy(data=focus_filter_50_utm)
    focus_filter_xarray_100_utm = focus_utm.copy(data=focus_filter_100_utm)
    focus_filter_xarray_150_utm = focus_utm.copy(data=focus_filter_150_utm)
    focus_filter_xarray_300_utm = focus_utm.copy(data=focus_filter_300_utm)
    focus_filter_xarray_450_utm = focus_utm.copy(data=focus_filter_450_utm)
    focus_filter_xarray_af_utm = focus_utm.copy(data=focus_filter_af_utm)

    # create xarray slopes
    dz_dlat_xar, dz_dlon_xar = focus_utm.differentiate(coord='y'), focus_utm.differentiate(coord='x')
    dz_dlat_filter_xar_50, dz_dlon_filter_xar_50 = focus_filter_xarray_50_utm.differentiate(coord='y'), focus_filter_xarray_50_utm.differentiate(coord='x')
    dz_dlat_filter_xar_100, dz_dlon_filter_xar_100 = focus_filter_xarray_100_utm.differentiate(coord='y'), focus_filter_xarray_100_utm.differentiate(coord='x')
    dz_dlat_filter_xar_150, dz_dlon_filter_xar_150 = focus_filter_xarray_150_utm.differentiate(coord='y'), focus_filter_xarray_150_utm.differentiate(coord='x')
    dz_dlat_filter_xar_300, dz_dlon_filter_xar_300  = focus_filter_xarray_300_utm.differentiate(coord='y'), focus_filter_xarray_300_utm.differentiate(coord='x')
    dz_dlat_filter_xar_450, dz_dlon_filter_xar_450  = focus_filter_xarray_450_utm.differentiate(coord='y'), focus_filter_xarray_450_utm.differentiate(coord='x')
    dz_dlat_filter_xar_af, dz_dlon_filter_xar_af = focus_filter_xarray_af_utm.differentiate(coord='y'), focus_filter_xarray_af_utm.differentiate(coord='x')

    # Calculate curvature and aspect using xrspatial
    curv_50 = xrspatial.curvature(focus_filter_xarray_50_utm)
    curv_300 = xrspatial.curvature(focus_filter_xarray_300_utm)
    curv_af = xrspatial.curvature(focus_filter_xarray_af_utm)
    aspect_50 = xrspatial.aspect(focus_filter_xarray_50_utm)
    aspect_300 = xrspatial.aspect(focus_filter_xarray_300_utm)
    aspect_af = xrspatial.aspect(focus_filter_xarray_af_utm)

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

    # Fill dataframe with elevation and slopes
    points_df['elevation'] = elevation_data
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
    points_df['slope_lat_gf450'] = slope_lat_data_filter_450
    points_df['slope_lon_gf450'] = slope_lon_data_filter_450
    points_df['slope_lat_gfa'] = slope_lat_data_filter_af
    points_df['slope_lon_gfa'] = slope_lon_data_filter_af
    points_df['curv_50'] = curv_data_50
    points_df['curv_300'] = curv_data_300
    points_df['curv_gfa'] = curv_data_af
    points_df['aspect_50'] = aspect_data_50
    points_df['aspect_300'] = aspect_data_300
    points_df['aspect_gfa'] = aspect_data_af

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

    tslope2 = time.time()
    tslope = tslope2-tslope1

    """ Calculate Farinotti ith_f """
    print(f"Calculating ith_f...")
    tfar1 = time.time()
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

    tfar2 = time.time()
    tfar = tfar2-tfar1

    """ Calculate distance_from_border """
    print(f"Calculating the distances using glacier geometries... ")
    tdist0 = time.time()

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
        geoseries_geometries_epsg = gpd.GeoSeries(cluster_exterior_ring + cluster_interior_rings, crs=glacier_epsg)

    # Get all generated points and create Geopandas geoseries and convert to UTM
    list_points = [Point(lon, lat) for (lon, lat) in zip(points_df['lons'], points_df['lats'])]
    geoseries_points_4326 = gpd.GeoSeries(list_points, crs="EPSG:4326")
    geoseries_points_epsg = geoseries_points_4326.to_crs(epsg=glacier_epsg)

    print(f"We have {len(geoseries_geometries_epsg)} geometries in the cluster")

    # Method that uses KDTree index (best method: found to be same as exact method and ultra fast)
    run_method_KDTree_index = True
    if run_method_KDTree_index:

        td1 = time.time()

        # Extract all coordinates of GeoSeries points
        points_coords_array = np.column_stack((geoseries_points_epsg.geometry.x, geoseries_points_epsg.geometry.y)) #(10000,2)

        # Extract all coordinates from the GeoSeries geometries
        geoms_coords_array = np.concatenate([np.array(geom.coords) for geom in geoseries_geometries_epsg.geometry])

        kdtree = KDTree(geoms_coords_array)

        # Perform nearest neighbor search for each point and calculate minimum distances
        distances, indices = kdtree.query(points_coords_array, k=len(geoseries_geometries_epsg))
        min_distances = np.min(distances, axis=1)

        min_distances /= 1000.

        # Retrieve the geometries corresponding to the indices

        td2 = time.time()
        print(f"Distances calculated with KDTree in {td2 - td1}")

    plot_minimum_distances = False
    if plot_minimum_distances:
        fig, ax = plt.subplots()
        #ax.plot(*gl_geom.exterior.xy, color='blue')
        ax.plot(*geoseries_geometries_epsg.loc[0].xy, lw=1, c='blue')  # first entry is outside border
        for geom in geoseries_geometries_epsg.loc[1:]:
            ax.plot(*geom.xy, lw=1, c='red')
        s1 = ax.scatter(x=points_coords_array[:,0], y=points_coords_array[:,1], s=10, c=min_distances, alpha=0.5, zorder=2)
        #s1 = ax.scatter(x=points_df['lons'], y=points_df['lats'], s=10, c=min_distances3, alpha=0.5, zorder=2)
        plt.colorbar(s1, ax=ax, label='Minimum Distance (km)')
        plt.show()

    # Method 3: geopandas spatial indexes (bad method and slow)
    run_method_geopandas_index = False
    if run_method_geopandas_index:
        min_distances = []
        sindex_id = geoseries_geometries_epsg.sindex
        for i, point_epsg in enumerate(geoseries_points_epsg):
            nearest_idx = sindex_id.nearest(point_epsg.bounds)
            nearest_geometries = geoseries_geometries_epsg.iloc[nearest_idx]
            min_distances_ = nearest_geometries.distance(point_epsg)
            min_idx = min_distances_.idxmin()
            min_dist = min_distances_.loc[min_idx]
            min_distances.append(min_dist / 1000.)

    # Method 3: vectorized version with CPU (exact method but slow)
    run_distances_with_geopandas_multicpu = False
    if run_distances_with_geopandas_multicpu:
        def calc_min_distance_to_multi_line(point, multi_line):
            min_dist = point.distance(multi_line)
            return min_dist

        td1 = time.time()
        multiline_geometries_epsg = MultiLineString(list(geoseries_geometries_epsg))
        args_list = [(point, multiline_geometries_epsg) for point in geoseries_points_epsg]
        min_distances = Parallel(n_jobs=-1)(delayed(calc_min_distance_to_multi_line)(*args) for args in args_list)
        min_distances = np.array(min_distances)
        min_distances /= 1000.  # km
        td2 = time.time()
        print(f"Distances using pandas distance and multicpu {td2 - td1}")

    # Method 4: not verctorized version (exact method but very slow)
    run_method_not_vectorized = False
    if run_method_not_vectorized:
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
            plot_calculate_distance = True
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

    points_df['dist_from_border_km_geom'] = min_distances
    tdist1 = time.time()
    tdist = tdist1 - tdist0
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
        im1 = dz_dlat_xar.plot(ax=ax1, cmap='gist_gray', vmin=np.nanmin(slope_lat_data),
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
    # At this stage any nan may be present in Millan features and Farinotti. I want to remove these points.
    #list_cols_for_nan_drop = ['ith_m', 'ith_f', 'vx','vy','vx_gf50', 'vx_gf100', 'vx_gf150', 'vx_gf300', 'vy_gf50',
    #                          'vy_gf100', 'vy_gf150', 'vy_gf300', 'dvx_dx', 'dvx_dy', 'dvy_dx', 'dvy_dy']
    list_cols_for_nan_drop = ['ith_f']

    # Data imputation for any nan survived in Millan velocities. Set all to zero.
    list_cols_for_nan_to_zero_replacement = ['vx', 'vy', 'vx_gf50', 'vx_gf100', 'vx_gf150',
                                             'vx_gf300', 'vy_gf50', 'vy_gf100', 'vy_gf150', 'vy_gf300',
                                             'dvx_dx', 'dvx_dy', 'dvy_dx', 'dvy_dy']
    points_df[list_cols_for_nan_to_zero_replacement] = points_df[list_cols_for_nan_to_zero_replacement].fillna(0)

    points_df = points_df.dropna(subset=list_cols_for_nan_drop)

    print(f"Important: we have generated {points_df['ith_m'].isna().sum()} points where Millan is nan.")

    # The only survived nans should be only in ith_m
    # Check for the presence of nans in the generated dataset.
    assert points_df.drop('ith_m', axis=1).isnull().any().any() == False, \
        "Nans in generated dataset other than in Millan velocity! Something to check."

    tend = time.time()

    print(f"************** TIMES **************")
    print(f"Geometries generation: {tgeometries:.2f}")
    print(f"Points generation: {tgenpoints:.2f}")
    print(f"Millan: {tmillan:.2f}")
    print(f"Slope: {tslope:.2f}")
    print(f"Farinotti: {tfar:.2f}")
    print(f"Distances: {tdist:.2f}")
    print(f"*******TOTAL FETCHING FEATURES in {tend - tin:.1f} sec *******")
    return points_df


if __name__ == "__main__":
    glacier_name = 'RGI60-11.00846'
    rgi = glacier_name[6:8]
    dem_rgi = fetch_dem(folder_mosaic=args.mosaic, rgi=rgi)
    generated_points_dataframe = populate_glacier_with_metadata(
                                            glacier_name=glacier_name,
                                            dem_rgi=dem_rgi,
                                            n=10000,
                                            seed=42)

# 'RGI60-07.00228' should be a multiplygon
# RGI60-11.00781 has only 1 neighbor
# RGI60-08.00001 has no Millan data
# RGI60-11.00846 has multiple intersects with neighbors
# RGI60-11.02774 has no neighbors
#RGI60-11.02884 has no neighbors
#'RGI60-11.01450' Aletsch # RGI60-11.02774
#RGI60-11.00590, RGI60-11.01894 no Millan data ?
#glacier_name = np.random.choice(RGI_burned)
#'RGI60-01.01701'
# RGI60-07.00832