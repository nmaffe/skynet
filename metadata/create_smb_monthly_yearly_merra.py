import os
import glob
import argparse
import math

import pandas as pd
import scipy
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import numpy as np
import xarray as xr
import rioxarray
import rasterio.features
import geopandas as gpd
import oggm
from oggm import utils
from pyproj import Transformer, CRS, Geod
import shapely
import shapely.geometry
from shapely.geometry import box, Point, Polygon, LineString, MultiLineString
from itertools import product
import matplotlib.pyplot as plt
from rgi_smb_elevation_funcs import smb_elev_functs

"""
- glc (M2TMNXGLC): Land ice surface diagnostic
- int (M2TMNXINT): Vertically integrated diagnostic
- flx (M2TMNXFLX): Surface flux diagnostic

PRECCU: Convective Rainfall                         [M2TMNXINT.5.12.4]
PRECLS: Large Scale Rainfall                        [M2TMNXINT.5.12.4]
PRECSN: Snowafall                                   [M2TMNXINT.5.12.4]
EVAP: Evaporation from Turbulence                   [M2TMNXINT.5.12.4]
RUNOFF: Glacier Meltwater Runoff                    [M2TMNXGLC.5.12.4]
PRECTOT: Total Precipitation                        [M2TMNXFLX.5.12.4]
PRECTOTCORR: Bias-corrected Total Precipitation     [M2TMNXFLX.5.12.4]
T2M: 2-meters temperature                           [M2IMNXASM.5.12.4]

Leggo su un articolo su TC che loro hanno usato:
PRECSNOLAND, PRECTOTLAND

Glacier Surface Mass calculation - options available:
1) SMB = PRECCU + PRECLS + PRECSN - EVAP - RUNOFF [Default]
2) SMB = PRECTOT - EVAP - RUNOFF
3) SMB = PRECTOTCORR - EVAP - RUNOFF [Recommended]

Options 1 and 2 are equivalent while option 3 used bias corrected
precipitation.
"""
cumM_dt_grace = pd.DataFrame({'rgi': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                              'mgrace': [-72.5,-15.1,-41.2,-31.8,np.nan,-15.9,-12.1,-1.7,-20.2,-2.9,-3.0,-1.8,
                                         -5.03,-5.2,-12.2,-1.6,-30.4,-0.7,np.nan]}).set_index('rgi')

def cumulative_sum(input_list):
    input_array = np.array(input_list)
    cumulative_sum_array = np.cumsum(input_array)
    return cumulative_sum_array.tolist()

def r2(yi, fi):
    ss_res = np.sum((yi - fi) ** 2)
    ss_tot = np.sum((yi - np.mean(yi)) ** 2)
    return 1 - (ss_res / ss_tot)


def gaussian_filter_with_nans(U, sigma, trunc=4.0):
    # Since the reprojection into utm leads to distortions (=nans) we need to take care of this during filtering
    # From David in https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    V = U.copy()
    V[np.isnan(U)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=[sigma, sigma], mode='nearest', truncate=trunc)
    W = np.ones_like(U)
    W[np.isnan(U)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=[sigma, sigma], mode='nearest', truncate=trunc)
    WW[WW == 0] = np.nan
    filtered_U = VV / WW
    return filtered_U

def pixel_area_km(latitude, resolution_lon_degrees, resolution_lat_degrees):
    # https://onlinelibrary.wiley.com/doi/epdf/10.1111/tgis.12636
    # A = R^2*delta_lon*(sin(lat2)-sin(lat1)
    R = 6378.1370  # Earth radius in kilometers
    #lat2 = math.radians(latitude+resolution_lat_degrees/2)
    #lat1 = math.radians(latitude-resolution_lat_degrees/2)
    #latitude = [90.0 89.5 89.0 88.5 88.0 87.5 .... 0.5 0.0 -0.5 -1.0 -1.5 ... -89.5 -90.0]
    latitude = np.array(latitude)
    lat2 = np.radians(abs(latitude))
    lat1 = np.radians(abs(latitude) - resolution_lat_degrees)
    pixel_area_km2 = R**2 * np.radians(resolution_lon_degrees) * (np.sin(lat2)-np.sin(lat1))
    return pixel_area_km2

parser = argparse.ArgumentParser()
parser.add_argument('--merra2_input_folder', type=str,default="/media/maffe/nvme/MERRA-2/")
parser.add_argument('--create_smb', type=int, default=0)
parser.add_argument('--create_rgi_areas_and_overlaps', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--load_smb_and_correct', type=int, default=0)
parser.add_argument('--downscale', type=int, default=0)
args = parser.parse_args()

utils.get_rgi_dir(version='62')  # setup oggm version


# Get MERRA-2 geopotential and calculate geopotential height
file_const =  "MERRA2_101.const_2d_asm_Nx.00000000.nc4"
merra_const = rioxarray.open_rasterio(f"{args.merra2_input_folder}{file_const}")
res_merra_lon, res_merra_lat = abs(merra_const.rio.resolution()[0]), abs(merra_const.rio.resolution()[1])
merra_const['FRLANDICE'] = merra_const['FRLANDICE'].where(merra_const['FRLANDICE'] != 0.0)
# Get some ingredients we need
merra_xy_shape = merra_const['FRLANDICE'].shape[-2:]
num_lat_pxs, num_lon_pxs = merra_xy_shape[0], merra_xy_shape[1]
transform = merra_const['FRLANDICE'].rio.transform()
xycoords = {'y': merra_const.coords['y'], 'x': merra_const.coords['x']}
xydims = {'y': merra_const.dims['y'], 'x': merra_const.dims['x']}

'''Calculate MERRA-2 dem'''
g = 9.80665
merra_const['H'] = merra_const['PHIS'] / g

'''MERRA's pixel areas in km2'''
pixel_areas_latitude = pixel_area_km(merra_const['FRLANDICE'].coords['y'], res_merra_lon, res_merra_lat)
pixel_areas_matrix = np.outer(pixel_areas_latitude, np.ones(num_lon_pxs))[np.newaxis, :, :]
merra_const['area'] = (('time', 'y', 'x'), pixel_areas_matrix)

'''Create Dataset for each rgi with maps of rgi areas, a binary version, and rgi interections with merra's masks.'''
if args.create_rgi_areas_and_overlaps:
    print(f"Begin creation of RGI areas and MERRA overlaps")
    #rgi_dim = np.array([7])
    rgi_dim = np.arange(1,20)
    coords_da = {'rgi': rgi_dim, 'y': merra_const.coords['y'], 'x': merra_const.coords['x']}
    coords_shape = (len(coords_da['rgi']), len(coords_da['y']), len(coords_da['x']))
    # Create an empty Dataset
    all_rgis_area_da = xr.Dataset(coords=coords_da)

    all_rgis_area_da['rgi_area'] = xr.DataArray(np.nan * np.empty(coords_shape), dims=('rgi', 'y', 'x'), coords=coords_da)
    all_rgis_area_da['rgi_binary_area'] = xr.DataArray(np.nan * np.empty(coords_shape), dims=('rgi', 'y', 'x'), coords=coords_da)
    all_rgis_area_da['rgi_binary_intersect'] = xr.DataArray(np.nan * np.empty(coords_shape), dims=('rgi', 'y', 'x'), coords=coords_da)

    for rgi in rgi_dim:

        rgi_polygons_raster_hits_np = np.zeros(merra_xy_shape)
        rgi_polygons_raster_area_np = np.zeros(merra_xy_shape)
        area_check = 0

        oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')  # get rgi region shp
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)  # get rgi dataset of glaciers
        area_rgi = oggm_rgi_glaciers['Area'].sum()  # Tot rgi Area [km2]
        gl_geoms = oggm_rgi_glaciers['geometry']
        gl_geoms_ext_gs = gl_geoms.exterior  # Geoseries
        gl_geoms_ext_gdf = gpd.GeoDataFrame(geometry=gl_geoms_ext_gs, crs="EPSG:4326")  # Geodataframe
        print(f"Rgi {rgi}, no. glaciers: {len(gl_geoms_ext_gdf)}")

        for n, (area, geom) in enumerate(zip(oggm_rgi_glaciers['Area'],
                                                   gl_geoms_ext_gdf['geometry'])):


            mask = rasterio.features.geometry_mask([geom], transform=transform, out_shape=merra_xy_shape,
                                                   all_touched=True, invert=True) #invert=True makes burned pixels=1
            no_px_burned = np.sum(mask)

            rgi_polygons_raster_hits_np += mask

            # Add the polygon area to the corresponding pixels in the occupied_area array
            if no_px_burned > 0:
                rgi_polygons_raster_area_np[mask] += area / no_px_burned

            plot_procedure = False
            if plot_procedure:
                fig, (ax1, ax2) = plt.subplots(2,1)
                ax1.imshow(mask)
                ax2.imshow(rgi_polygons_raster_area_np)
                plt.show()

            area_check += area

        # Create a new xarray DataArray with the calculated rgi area values
        rgi_area_ar = xr.DataArray(data=rgi_polygons_raster_area_np, coords=xycoords, dims=xydims, name="rgi_area")
        # A binary version could be useful somewhere
        rgi_binary_np = np.where(rgi_polygons_raster_area_np > 0, 1, np.nan)
        rgi_binary_ar = xr.where(rgi_area_ar>0, 1, np.nan)

        print(f"Area rgi: {area_rgi} km2")
        print(f"Area check: {area_check} km2")
        print(f"Area check2: {np.nansum(rgi_polygons_raster_area_np)} km2")

        # Calculate the binary intersection between rgi and merra masks
        rgi_binary_intersect = rgi_binary_ar.where(merra_const['FRLANDICE'].values.squeeze()>0).rename("intersection_rgi_merra")

        intersection_rgi_merra = np.where((rgi_binary_np > 0) & (merra_const['FRLANDICE'].values.squeeze() > 0), 1, np.nan)
        print(f"In rgi {rgi} we have {np.nansum(rgi_binary_np)} rgi pixels of which"
              f" {np.nansum(intersection_rgi_merra)} intersections with merra")

        # Fill xarray dataset
        all_rgis_area_da['rgi_area'].loc[{'rgi': rgi}] = rgi_polygons_raster_area_np
        all_rgis_area_da['rgi_binary_area'].loc[{'rgi': rgi}] = rgi_binary_np
        all_rgis_area_da['rgi_binary_intersect'].loc[{'rgi': rgi}] = intersection_rgi_merra

        #print(rgi_binary_intersect.values.shape, intersection_rgi_merra.shape)
        #print(np.array_equal(rgi_binary_intersect.values, intersection_rgi_merra))
        # print(all_rgis_area_da['rgi_area'].loc[{'rgi': 7}])
        #all_rgis_area_da['rgi_binary_area'].loc[{'rgi': rgi}].plot()
        #rgi_binary_intersect.plot()
        #plt.show()
        #exit()

        plot_polygons_raster = False
        if plot_polygons_raster:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12,8))
            im1 = (merra_const['FRLANDICE'] * merra_const['area']).plot(ax=ax1)
            #im2 = (rgi_area_ar * rgi_binary_ar).plot(ax=ax2)
            #im2 = rgi_area_ar.plot(ax=ax2)
            im2 = all_rgis_area_da['rgi_area'].loc[{'rgi': rgi}].plot(ax=ax2)
            im3 = all_rgis_area_da['rgi_binary_intersect'].loc[{'rgi': rgi}].plot(ax=ax3)
            #im3 = (rgi_binary_intersect).plot(ax=ax3)
            im4 = (rgi_binary_intersect*(merra_const['FRLANDICE'][0] * merra_const['area'][0])/rgi_area_ar).plot(ax=ax4)
            min_lon = gl_geoms_ext_gdf.total_bounds[0]
            min_lat = gl_geoms_ext_gdf.total_bounds[1]
            max_lon = gl_geoms_ext_gdf.total_bounds[2]
            max_lat = gl_geoms_ext_gdf.total_bounds[3]
            for ax in [ax1, ax2, ax3, ax4]:
                gl_geoms_ext_gdf.plot(ax=ax, color='r')
                ax.set_xlim(min_lon-1, max_lon+1)
                ax.set_ylim(min_lat-1, max_lat+1)
            plt.tight_layout()
            plt.show()

    if args.save:
        file_out = "rgi_and_merra_masks_and_areas.nc"
        all_rgis_area_da.rio.write_crs("EPSG:4326", inplace=True)
        all_rgis_area_da = all_rgis_area_da.rio.write_transform(transform) # important
        all_rgis_area_da.to_netcdf(f"{args.merra2_input_folder}{file_out}")
        print(f"{file_out} saved. Exit.")

    exit()

''' Create monthly smb maps '''
if args.create_smb:

    time_series_smb_month = []

    for i, (year, month) in enumerate(product(range(2002, 2020), range(1,13))):
        t = year + month / 12
        print(f'Year: {year}, Month: {month}, time {t}')

        file_int =  glob.glob(f"{args.merra2_input_folder}{year}/int/MERRA2_*.tavgM_2d_int_Nx.{year}{month:02}.nc4")[0]
        file_glc =  glob.glob(f"{args.merra2_input_folder}{year}/glc/MERRA2_*.tavgM_2d_glc_Nx.{year}{month:02}_ML.nc4")[0]
        file_flx =  glob.glob(f"{args.merra2_input_folder}{year}/flx/MERRA2_*.tavgM_2d_flx_Nx.{year}{month:02}.nc4")[0]
        file_lnd =  glob.glob(f"{args.merra2_input_folder}{year}/lnd/MERRA2_*.tavgM_2d_lnd_Nx.{year}{month:02}.nc4")[0]

        # Import files and create Datasets
        ds_glc = rioxarray.open_rasterio(file_glc, crs="EPSG:4326")
        ds_int = rioxarray.open_rasterio(file_int, crs="EPSG:4326")
        #ds_flx = rioxarray.open_rasterio(file_flx, crs="EPSG:4326")
        #ds_lnd = rioxarray.open_rasterio(file_lnd, crs="EPSG:4326")

        list_datasets = [ds_glc, ds_int,] #ds_flx, ds_lnd
        for raster in list_datasets: raster.rio.write_crs("EPSG:4326", inplace=True)

        # Create DataArray and replace fill values with nans
        ds_int['PRECCU'] = ds_int['PRECCU'].where(ds_int['PRECCU'] != ds_int['PRECCU'].rio.nodata)
        ds_int['PRECLS'] = ds_int['PRECLS'].where(ds_int['PRECLS'] != ds_int['PRECLS'].rio.nodata)
        ds_int['PRECSN'] = ds_int['PRECSN'].where(ds_int['PRECSN'] != ds_int['PRECSN'].rio.nodata)
        ds_int['EVAP'] = ds_int['EVAP'].where(ds_int['EVAP'] != ds_int['EVAP'].rio.nodata)
        ds_glc['RUNOFF'] = ds_glc['RUNOFF'].where(ds_glc['RUNOFF'] != ds_glc['RUNOFF'].rio.nodata)
        ds_glc['RUNOFF_ML'] = ds_glc['RUNOFF_ML'].where(ds_glc['RUNOFF_ML'] != ds_glc['RUNOFF_ML'].rio.nodata)
        #ds_flx['PRECTOT'] = ds_flx['PRECTOT'].where(ds_flx['PRECTOT'] != ds_flx['PRECTOT'].rio.nodata)
        #ds_flx['PRECTOTCORR'] = ds_flx['PRECTOTCORR'].where(ds_flx['PRECTOTCORR'] != ds_flx['PRECTOTCORR'].rio.nodata)
        #ds_lnd['PRECTOTLAND'] = ds_lnd['PRECTOTLAND'].where(ds_lnd['PRECTOTLAND'] != ds_lnd['PRECTOTLAND'].rio.nodata)

        ds_int['PRECCU'].attrs['_FillValue'] = np.nan
        ds_int['PRECLS'].attrs['_FillValue'] = np.nan
        ds_int['PRECSN'].attrs['_FillValue'] = np.nan
        ds_int['EVAP'].attrs['_FillValue'] = np.nan
        ds_glc['RUNOFF'].attrs['_FillValue'] = np.nan
        ds_glc['RUNOFF_ML'].attrs['_FillValue'] = np.nan
        #ds_flx['PRECTOT'].attrs['_FillValue'] = np.nan
        #ds_flx['PRECTOTCORR'].attrs['_FillValue'] = np.nan
        #ds_lnd['PRECTOTLAND'].attrs['_FillValue'] = np.nan

        # There is an hour mismatch between the time variables. Let's use int time variable for glc
        ds_glc['time'] = ds_int['time']

        # Calculate SMB DataArray
        #SMB_month = (ds_flx['PRECTOT'].values - ds_int['EVAP'].values- ds_glc['RUNOFF'].values)
        #SMB_month = (ds_lnd['PRECTOTLAND'].values - ds_int['EVAP'].values - ds_glc['RUNOFF'].values)
        SMB_month_vals = (ds_int['PRECCU'].values + ds_int['PRECLS'].values + ds_int['PRECSN'].values
                     - ds_int['EVAP'].values - ds_glc['RUNOFF'].values)
        SMB_month_vals_ML = (ds_int['PRECCU'].values + ds_int['PRECLS'].values + ds_int['PRECSN'].values
                     - ds_int['EVAP'].values - ds_glc['RUNOFF_ML'].values)
        # * (merra_const['FRLANDICE'].values * rgi_polygons_raster_ar.values))


        # create SMB Dataset
        ds_smb_month = xr.Dataset(coords=ds_int.coords)
        ds_smb_month['SMB'] = (('time', 'y', 'x'), SMB_month_vals)
        ds_smb_month['SMB_ML'] = (('time', 'y', 'x'), SMB_month_vals_ML)

        # Append to list
        time_series_smb_month.append(ds_smb_month)

        # Plot
        plot = False
        if plot:
            fig, axs = plt.subplots(3,2, figsize=(10,6))
            ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
            im1 = ds_int['PRECCU'].plot(ax=ax1)
            im2 = ds_int['PRECLS'].plot(ax=ax2)
            im3 = ds_int['PRECSN'].plot(ax=ax3)
            im4 = ds_int['EVAP'].plot(ax=ax4, cmap='seismic')
            im5 = ds_glc['RUNOFF'].plot(ax=ax5)
            im6 = ds_smb_month['SMB'].plot(ax=ax6, cmap='seismic')
            for ax in axs.flatten():
                gl_geoms_ext_gdf.plot(ax=ax, color='r')

            plt.tight_layout()
            plt.show()

    combined_dataset = xr.concat(time_series_smb_month, dim='time')

    if args.save:
        file_out = "smb_series.nc"
        combined_dataset.rio.write_crs("EPSG:4326", inplace=True)
        combined_dataset.to_netcdf(f"{args.merra2_input_folder}{file_out}")
        print(f"{file_out} saved. Exit.")

    exit()

if args.load_smb_and_correct:

    ''' Create monthly time series of cumulative SMB'''
    # We have the following:
    # merra_const['FRLANDICE']: merra glacier masks in ice percentage
    # merra_const['area']: merra pixel areas
    # rgi_area_ar: rgi pixel areas
    # rgi_binary_ar: binary rgi pixel areas
    # rgi_binary_intersect: binary intersection between rgi and merra masks

    t_i, t_f = 2002, 2020
    secs_in_month = 2628288

    smb_da = rioxarray.open_rasterio(f"{args.merra2_input_folder}smb_series.nc")
    all_rgis_area_da = rioxarray.open_rasterio(f"{args.merra2_input_folder}rgi_and_merra_masks_and_areas.nc")
    # - all_rgis_area_da['rgi_area'] has rgi areas for each rgi
    # - all_rgis_area_da['rgi_binary_area'] has rgi binary areas for each rgi
    # - all_rgis_area_da['rgi_binary_intersect'] has rgi binary intersections with merra

    # Questo sara il nuovo dataset di smb corretti
    smb_corr_da = smb_da.copy().drop_vars(['SMB', 'SMB_ML'])
    smb_corr_da['SMB'] = (('time', 'y', 'x'), np.zeros(smb_da['SMB'].shape))

    for rgi in [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]:

        rgi_area_ar = all_rgis_area_da['rgi_area'].loc[{'rgi': rgi}]
        rgi_binary_ar = all_rgis_area_da['rgi_binary_area'].loc[{'rgi': rgi}]
        rgi_binary_intersect = all_rgis_area_da['rgi_binary_intersect'].loc[{'rgi': rgi}]

        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #rgi_binary_ar.plot(ax=ax1, cmap='binary')
        #rgi_binary_intersect.plot(ax=ax2, cmap='binary')
        #plt.show()

        # Create regional monthly smb maps and mass
        time_series_rgi_mass = []

        for n, time in enumerate(smb_da['time'].values):

            # Extract the data for the current time step
            smb_monthly = smb_da.isel(time=n)

            # Calculate rgi smb
            # in the intersect we want to keep 'SMB'. Elsewhere we augment with 'SMB_ML'
            smb_rgi_monthML = smb_monthly['SMB_ML'] * rgi_binary_ar
            smb_rgi_monthMERRA = smb_monthly['SMB'] * rgi_binary_intersect
            smb_rgi_month = xr.where(smb_rgi_monthMERRA.isnull(), smb_rgi_monthML, smb_rgi_monthMERRA)

            plot_sth = False
            if plot_sth:
                fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                #rgi_binary_ar.plot(ax=ax1, cmap='binary')
                #rgi_binary_intersect.plot(ax=ax2, cmap='binary')
                #merra_const['FRLANDICE'].plot(ax=ax3, cmap='viridis')
                #smb_rgi_month.plot(ax=ax4, cmap='binary')
                smb_monthly['SMB_ML'].plot(ax=ax1, cmap='viridis')
                smb_monthly['SMB'].plot(ax=ax2, cmap='viridis')
                smb_rgi_month.plot(ax=ax3, cmap='viridis')
                plt.show()


            # Weight the smb for fraction of ice
            #smb_rgi_month2 = smb_rgi_month / (merra_const['FRLANDICE'].values.squeeze() * merra_const['area'].values.squeeze()) * rgi_area_ar.values

            plot_monthly_smb = False
            if plot_monthly_smb:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                im1 = smb_rgi_month.plot(ax=ax1)
                im2 = smb_rgi_month.plot(ax=ax2)
                im3 = (merra_const['FRLANDICE'] * merra_const['area']).plot(ax=ax3)
                im4 = (rgi_area_ar / (merra_const['FRLANDICE'] * merra_const['area'])).plot(ax=ax4)
                for ax in [ax1, ax2, ax3, ax4]:
                    gl_geoms_ext_gdf.plot(ax=ax, color='r')
                plt.tight_layout()
                plt.show()



            # tot_smb_rgi_month = float(smb_rgi_month2['SMB'].sum(skipna=True)) # kg/m2*s
            # tot_smb_rgi_month = float((smb_rgi_month2 * merra_const['FRLANDICE'].values.squeeze() * merra_pixel_areas.values.squeeze()).sum(skipna=True))
            mass_per_sec_rgi_month = float((smb_rgi_month * rgi_area_ar).sum(skipna=True)) # kg/s

            mass_rgi_month = secs_in_month * mass_per_sec_rgi_month * 1e6 #* area_rgi  # kg
            mass_rgi_month /= 1e12 #Gt
            print(f"rgi {rgi} {n} {mass_rgi_month}")

            time_series_rgi_mass.append(mass_rgi_month)


        # Calculate cumulative mass
        time_series_rgi_cummass = cumulative_sum(time_series_rgi_mass)

        mean_cummass = np.mean(time_series_rgi_cummass)

        # zero mean the cumulative series
        time_series_rgi_cummass_0m = time_series_rgi_cummass - mean_cummass

        num_months_series = len(time_series_rgi_mass)
        x_time_years = [t_i+ m/12 for m in range(num_months_series)]

        # Fit
        m0, q0, r0, p0, se0 = stats.linregress(x=x_time_years, y=time_series_rgi_cummass_0m)
        print(f"slope {m0} intercept {q0}")

        # Get GRACE dcumM/dt slope
        m_grace = cumM_dt_grace.loc[rgi, 'mgrace']
        print(f"Forcing the slope to be that of grace: {m_grace}")

        # Zero-mean corrected cumulative series
        time_series_rgi_cummass_corr_0m = time_series_rgi_cummass_0m / (m0/m_grace)
        m1, q1, r1, p1, se1 = stats.linregress(x=x_time_years, y=time_series_rgi_cummass_corr_0m)
        print(np.mean(time_series_rgi_cummass_0m), np.mean(time_series_rgi_cummass_corr_0m))

        # Original corrected cumulative series (not zero-mean)
        shift = time_series_rgi_cummass_corr_0m[0] - time_series_rgi_cummass[0]
        time_series_rgi_cummass_corr = time_series_rgi_cummass_corr_0m - shift

        # Original corrected mass series (given the cumulative time_series_rgi_cummass_corr, calculate the series)
        time_series_rgi_mass_corr = np.ediff1d(time_series_rgi_cummass_corr, to_begin=time_series_rgi_cummass_corr[0])
        # check_cum = cumulative_sum(time_series_rgi_mass_corr) # this is a check

        plot_series = False
        if plot_series:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.plot(x_time_years, time_series_rgi_mass, 'o-')
            ax1.plot(x_time_years, time_series_rgi_mass_corr, '*-', c='k')
            ax2.plot(x_time_years, time_series_rgi_cummass, '-')
            ax2.plot(x_time_years, time_series_rgi_cummass_0m, '-', c='tab:blue')
            ax2.plot(x_time_years, time_series_rgi_cummass_corr_0m, '--', c='k')
            ax2.plot(x_time_years, time_series_rgi_cummass_corr, '--', c='k')
            ax2.axline(xy1=(t_i, m0*t_i+q0), slope=m0, c='tab:blue', lw=2, label=f"$y = {m0:.1f}x {q0:+.1f}$")
            ax2.axline(xy1=(t_i, m1*t_i+q1), slope=m1, c='gray', lw=2, label=f"$y = {m1:.1f}x {q1:+.1f}$")
            ax2.legend()
            ax1.set_ylabel(f"rgi {rgi} monthly mass (Gt)")
            plt.show()

        # Now I have to loop again over each month and correct the smb maps
        for n, time in enumerate(smb_da['time'].values):
            smb_monthly = smb_da.isel(time=n)

            # Calculate rgi smb
            # in the intersect we want to keep 'SMB'. Elsewhere we augment with 'SMB_ML'
            smb_rgi_monthML = smb_monthly['SMB_ML'] * rgi_binary_ar
            smb_rgi_monthMERRA = smb_monthly['SMB'] * rgi_binary_intersect
            smb_rgi_month = xr.where(smb_rgi_monthMERRA.isnull(), smb_rgi_monthML, smb_rgi_monthMERRA)

            old_mass_rgi_month = time_series_rgi_mass[n]
            correct_rgi_month = time_series_rgi_mass_corr[n]

            # smb_rgi_month should be modified to that the mass would match correct_rgi_month
            corr_month = correct_rgi_month / old_mass_rgi_month
            # We assume we can factorize the smb map as smb_corr = A * smb_old
            smb_rgi_month_corr = corr_month * smb_rgi_month

            #check that the new mass is correct
            #check_mass_per_sec_rgi_month = float((smb_rgi_month_corr * rgi_area_ar).sum(skipna=True))  # kg/s
            #check_mass_rgi_month = secs_in_month * check_mass_per_sec_rgi_month * 1e6  # * area_rgi  # kg
            #check_mass_rgi_month /= 1e12  # Gt
            # print(n, old_mass_rgi_month, correct_rgi_month, check_mass_rgi_month, correct_rgi_month - check_mass_rgi_month)

            # Fill corrected smb map for specific month and rgi regions
            smb_corr_da.isel(time=n)['SMB'] += smb_rgi_month_corr.fillna(0)
            #smb_corr_da.isel(time=n)['SMB'].plot(cmap='binary')
            #plt.show()


    #for n, time in enumerate(smb_corr_da['time'].values):
    #    smb_corr_da.isel(time=n)['SMB'].plot(cmap='bwr')
    #    plt.show()

    # 2002-2020 time averaged smb
    smb_corr_mean_da = smb_corr_da.mean(dim='time')

    if args.save:
        file_out = "smb_corrected_mean_2002_2019.nc"
        smb_corr_mean_da.rio.write_crs("EPSG:4326", inplace=True)
        smb_corr_mean_da = smb_corr_mean_da.rio.write_transform(transform)  # important
        smb_corr_mean_da.to_netcdf(f"{args.merra2_input_folder}{file_out}")
        print(f"{file_out} saved. Exit.")
        exit()

if args.downscale:

    smb_corr_mean_da = rioxarray.open_rasterio(f"{args.merra2_input_folder}smb_corrected_mean_2002_2019.nc") # dataarray
    all_rgis_area_da = rioxarray.open_rasterio(f"{args.merra2_input_folder}rgi_and_merra_masks_and_areas.nc")

    # Mask
    rgi = 17
    mask_downscaling = all_rgis_area_da['rgi_binary_area'].sel(rgi=rgi).fillna(0) #+ all_rgis_area_da['rgi_binary_area'].sel(rgi=14).fillna(0)

    # mask 1
    #mask1_downscaling = mask_downscaling.where(((mask_downscaling['x'] > 150.0) & (mask_downscaling['y'] < 67.0))
    #                                          | ((mask_downscaling['x'] < -170.0) & (mask_downscaling['y'] < 68.0)) , 0)
    # mask 2
    #mask2_downscaling = mask_downscaling.where((mask_downscaling['y'] > 70.5), 0)
    # mask 3
    #mask3_downscaling = mask_downscaling.where(((mask_downscaling['x'] > 123.0) & (mask_downscaling['x'] < 150.0))
    #                                          & ((mask_downscaling['y'] > 59.0) & (mask_downscaling['y'] < 71.0)) , 0)
    # mask 4
    #mask4_downscaling = mask_downscaling.where(((mask_downscaling['x'] > 55.0) & (mask_downscaling['x'] < 98.0))
    #                                          & ((mask_downscaling['y'] > 62.0) & (mask_downscaling['y'] < 71.0)) , 0)
    # mask 5
    #mask5_downscaling = mask_downscaling.where(((mask_downscaling['x'] > 83.0) & (mask_downscaling['x'] < 121.0))
    #                                          & ((mask_downscaling['y'] > 44.0) & (mask_downscaling['y'] < 60.0)) , 0)

    # rgi 16
    # mask africa and indonesia
    #mask1_downscaling = mask_downscaling.where(mask_downscaling['x'] > -20, 0)

    # northern part
    #mask2_downscaling = mask_downscaling.where(((mask_downscaling['y'] > -4.0) & (mask_downscaling['x'] < -60.0)), 0)

    # south west part
    #mask3_downscaling = mask_downscaling.where(((mask_downscaling['y'] < -4.0) & (mask_downscaling['x'] < -74.0)), 0)

    # south est part (constant fit)
    #mask4_downscaling = mask_downscaling.where(((mask_downscaling['y'] < -4.0) &
    #                                            (mask_downscaling['x'] > -74.0) & (mask_downscaling['x'] < -60.0)), 0)

    # rgi 17
    mask1_downscaling = mask_downscaling.where(mask_downscaling['y'] <= -52., 0)
    mask2_downscaling = mask_downscaling.where(mask_downscaling['y'] > -52., 0)

    # Force to use some sub regional mask
    mask_downscaling = (mask1_downscaling)

    mask_downscaling.plot(cmap='binary')
    plt.show()

    # Masked elevation and smb maps
    h_mask = merra_const['H'].where(mask_downscaling.values>0)
    smb_mask = smb_corr_mean_da.where(mask_downscaling.values>0)

    # if i want to separately see different rgis
    #mask2 = all_rgis_area_da['rgi_binary_area'].sel(rgi=13).fillna(0)
    #h_mask2 = merra_const['H'].where(mask2.values>0).values.flatten()
    #smb_mask2 = smb_corr_mean_da.where(mask2.values > 0).values.flatten()

    # Masked elevation and smb values
    h_mask_vals = h_mask.values.flatten()
    smb_mask_vals = smb_mask.values.flatten()

    # Get rid of nans
    mask_non_nans = ~np.isnan(h_mask_vals) & ~np.isnan(smb_mask_vals)
    h_mask_vals = h_mask_vals[mask_non_nans]
    smb_mask_vals = smb_mask_vals[mask_non_nans]

    # Filter outliers
    if rgi==13:
        mask_outliers = ~((h_mask_vals>3500) & (smb_mask_vals<-8.5e-6)) & (smb_mask_vals<-2.0e-6)
        h_mask_vals, smb_mask_vals = h_mask_vals[mask_outliers], smb_mask_vals[mask_outliers]
    if rgi==14:
        mask_outliers = ~((h_mask_vals>2500) & (smb_mask_vals<-1.8e-5))
        h_mask_vals, smb_mask_vals = h_mask_vals[mask_outliers], smb_mask_vals[mask_outliers]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
    smb_corr_mean_da.plot(ax=ax1, cmap='bwr')
    mask_downscaling.plot(ax=ax2, cmap='binary')
    h_mask.plot(ax=ax3, cmap='viridis')
    smb_mask.plot(ax=ax4, cmap='viridis')
    ax5.scatter(x=h_mask_vals, y=smb_mask_vals)
    plt.show()

    def func_fit_rgi12(x, a, c, p):
        return a / (c + x)**p
    def func_fit(x, m, q):
        return m * x + q


    popt, pcov = curve_fit(func_fit, h_mask_vals, smb_mask_vals, bounds=([-np.inf, -np.inf], [np.inf, np.inf]))

    print(f"Rgi: {rgi}")
    print(f" Parameters: {popt} pm {np.sqrt(np.diag(pcov))}")
    print(f"Coefficient of determination r2 = {r2(smb_mask_vals, func_fit(h_mask_vals, *popt))}")

    #y_check = smb_elev_functs(rgi=rgi, elev=h_mask_vals, lon=85, lat=-10)

    fig, ax = plt.subplots()
    ax.scatter(x=h_mask_vals, y=smb_mask_vals)
    ax.plot(np.sort(h_mask_vals), func_fit(np.sort(h_mask_vals), *popt), 'r--', lw=2,label=f"{popt}") #label=f"a:{popt[0]:.3f}, b:{popt[1]:.3f}, "
                                                                                  #      f"c: {popt[2]:.3f}, p:{popt[3]:.3f}")
    #ax.scatter(x=h_mask_vals, y=y_check, c='k', s=4)
    #ax.scatter(x=h_mask2, y=smb_mask2, c='g')
    ax.legend()
    plt.show()





