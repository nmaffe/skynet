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
from pyproj import Proj, Transformer
from math import radians, cos, sin, asin, sqrt, floor
import utm

"""
This program generates glacier metadata at some random locations inside the glacier geometry. 

Input: glacier name (RGIId), how many points you want to generate. 
Output: pandas dataframe with features calculated for each generated point. 

Note: the points are generated inside the glacier but outside nunataks (there is a check for this)

Note: Millan and Farinotti products needs to be interpolated. Interpolation close to the borders may result in nans. 
The interpolation method="nearest" yields much less nans close to borders if compared to linear
interpolation and therefore is chosen. 

Note that Farinotti interpolation ith_f may result in nan when generated point too close to the border.

Note the following policy for Millan special cases to produce vx, vy, v, ith_m:
    1) There is no Millan data for such glacier. I force fake vy=vy=v=0.0 and keep ith_m=nan. 
    2) In case the interpolation of Millan's fields yields nan because points are either too close to the margins. 
    I keep the nans that will be however removed before returning the dataset.   
"""
# todo: question remains open on if and how to smooth millan, farinotti and slope fiels before interpolation
# todo: so far I use only neighboring pixels but Eric suggests to account for a wider window.

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


class CFG:
    # I need to reconstruct these features for each point created inside the glacier polygon
    features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
     'dist_from_border_km_geom', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
     'elevation_from_zmin', 'RGI', 'ith_m', 'ith_f']


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
    points_df = pd.DataFrame(columns=CFG.features + ['lons', 'lats', 'nunataks'])

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

    # Fill lats, lons and nunataks
    points_df['lats'] = points['lats']
    points_df['lons'] = points['lons']
    points_df['nunataks'] = points['nunataks']
    if (points_df['nunataks'].sum() != 0):
        print(f"The generation pipeline has produced n. {points_df['nunataks'].sum()} points inside nunataks")
        raise ValueError


    # Let's start filling the other features
    points_df['RGI'] = rgi
    points_df['Area'] = gl_df['Area'].item()
    points_df['Zmin'] = gl_df['Zmin'].item()
    points_df['Zmax'] = gl_df['Zmax'].item()
    points_df['Zmed'] = gl_df['Zmed'].item()
    points_df['Slope'] = gl_df['Slope'].item()
    points_df['Lmax'] = gl_df['Lmax'].item()

    """ Add Slopes and Elevation """
    print(f"Calculating slopes and elevations...")
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

        # Interpolate
        # Note that points generated very very close
        # to the glaciers boundaries may result in nan interpolation. This should be removed at the end.
        vx_data = focus_vx.interp(y=xarray.DataArray(lats_crs), x=xarray.DataArray(lons_crs),
                                  method="nearest").data.squeeze()
        vy_data = focus_vy.interp(y=xarray.DataArray(lats_crs), x=xarray.DataArray(lons_crs),
                                  method="nearest").data.squeeze()
        ith_data = focus_ith.interp(y=xarray.DataArray(lats_crs), x=xarray.DataArray(lons_crs),
                                    method="nearest").data.squeeze()
        print(
            f"From Millan vx, vy, ith interpolations we have generated {np.isnan(vx_data).sum()}/{np.isnan(vy_data).sum()}/{np.isnan(ith_data).sum()} nans.")

        # Fill dataframe with vx, vy, ith_m
        points_df['vx'] = vx_data  # note this may contain nans from interpolation at the margin/inside nunatak
        points_df['vy'] = vy_data  # note this may contain nans from interpolation at the margin/inside nunatak
        points_df['ith_m'] = ith_data  # note this may contain nans from interpolation at the margin/inside nunatak
        points_df['v'] = np.sqrt(points_df['vx'] ** 2 + points_df['vy'] ** 2)  # note this may contain nans
        no_millan_data = False
    except:
        print(f"No Millan data can be found for rgi {rgi} glacier {glacier_name}")
        no_millan_data = True
        for col in ['vx','vy', 'v']: # Fill Millan velocities with zero (keep ith_m as nan)
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
        assert epsg==glacier_epsg, f"Inconsistency found between point espg {epsg} and glacier center epsg {glacier_epsg}."

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
    print(f"Finished distance calculations.")

    # Show the result
    show_glacier_with_produced_points = True
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

        plt.tight_layout()
        plt.show()


    """ Cleaning the produced dataset """
    # At this stage any nan may be present in vx, vy, v, ith_m, ith_f. Remove those points.
    points_df = points_df.dropna(subset=['vx', 'vy', 'v', 'ith_m', 'ith_f'])
    print(points_df.T)
    print(f"Are there nans ? {points_df.isnull().any().any()}")
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

generated_points_dataframe = populate_glacier_with_metadata(glacier_name='RGI60-11.01450', n=200)

# 'RGI60-07.00228' should be a multiplygon
# RGI60-11.00781 has only 1 neighbor
# RGI60-08.00001 has no Millan data
# RGI60-11.00846 has multiple intersects with neighbors
# RGI60-11.02774 has no neighbors
#RGI60-11.02884 has no neighbors
#'RGI60-11.01450' Aletsch # RGI60-11.02774
#RGI60-11.00590, RGI60-11.01894 no Millan data ?
#glacier_name = np.random.choice(RGI_burned)