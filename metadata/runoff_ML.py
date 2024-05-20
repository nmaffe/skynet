import os
import glob
import argparse
from scipy import stats
import pandas as pd
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
from matplotlib.colors import LogNorm
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


parser = argparse.ArgumentParser()
parser.add_argument('--merra2_input_folder', type=str,default="/media/maffe/nvme/MERRA-2/")
args = parser.parse_args()

file_const =  "MERRA2_101.const_2d_asm_Nx.00000000.nc4"
merra_const = rioxarray.open_rasterio(f"{args.merra2_input_folder}{file_const}")
merra_const['H'] = merra_const['PHIS'] / 9.80665

class CFG:
    target = 'runoff'
    cols_train = ['month', 'year', 'h', 't2m', 'sw']
    N_feats = len(cols_train)
    # model = xgb.XGBRegressor()
    model = lgb.LGBMRegressor()
    min_t2m = 250.
    min_runoff = 0.0
    force_runoff_value = 1.e-9

def build_dataset(year_in=2002, year_fin=2020):
    dataframe = None

    for i, (year, month) in enumerate(product(range(year_in, year_fin), range(1,13))):
        t = year + month / 12
        print(f'Year: {year}, Month: {month}, time {t}')

        file_int = glob.glob(f"{args.merra2_input_folder}{year}/int/MERRA2_*.tavgM_2d_int_Nx.{year}{month:02}.nc4")[0]
        file_glc =  glob.glob(f"{args.merra2_input_folder}{year}/glc/MERRA2_*.tavgM_2d_glc_Nx.{year}{month:02}.nc4")[0]
        file_asm =  glob.glob(f"{args.merra2_input_folder}{year}/asm/MERRA2_*.instM_2d_asm_Nx.{year}{month:02}.nc4")[0]

        # Import files and create Datasets
        ds_int = rioxarray.open_rasterio(file_int, crs="EPSG:4326")
        ds_glc = rioxarray.open_rasterio(file_glc, crs="EPSG:4326")
        ds_asm = rioxarray.open_rasterio(file_asm, crs="EPSG:4326")

        list_datasets = [ds_int, ds_glc, ds_asm]
        for raster in list_datasets: raster.rio.write_crs("EPSG:4326", inplace=True)

        # Create DataArray and replace fill values with nans
        ds_glc['RUNOFF'] = ds_glc['RUNOFF'].where(ds_glc['RUNOFF'] != ds_glc['RUNOFF'].rio.nodata)
        ds_asm['T2M'] = ds_asm['T2M'].where(ds_asm['T2M'] != ds_asm['T2M'].rio.nodata)
        ds_int['SWNETSRF'] = ds_int['SWNETSRF'].where(ds_int['SWNETSRF'] != ds_int['SWNETSRF'].rio.nodata)

        #ds_int['SWNETSRF'].plot() # SHORTWAVE
        #(ds_int['LWGNET']+ds_int['SWNETSRF']).plot()
        #plt.show()

        ds_glc['time'] = ds_asm['time']
        ds_int['time'] = ds_asm['time']

        # Get mask of valid runoff values
        mask_non_null = ds_glc['RUNOFF'].notnull() # (1, 361, 576)

        # Get the indexes that correspond to valid runoff values (3 indexes arrays: time, lat, lon)
        indices = np.where(mask_non_null)
        indices_lat = indices[1]
        indices_lon = indices[2]

        latitude_non_null = ds_glc['y'].values[indices_lat]
        longitude_non_null = ds_glc['x'].values[indices_lon]

        runoff_non_null = ds_glc['RUNOFF'].values[indices]
        t2m_non_null = ds_asm['T2M'].values[indices]
        alt_non_null = merra_const['H'].values[indices]
        swnetsrf_non_null = ds_int['SWNETSRF'].values[indices]

        plt_scatter = False
        if plt_scatter:
            fig, (ax1, ax2) = plt.subplots(1,2)
            s1 = ax1.scatter(x=t2m_non_null, y=runoff_non_null, c=latitude_non_null,
                            s= np.interp(swnetsrf_non_null, (min(swnetsrf_non_null), max(swnetsrf_non_null)), (10, 100)),
                            alpha=.3) # t2m_non_null
            s2 = ax2.scatter(x=t2m_non_null, y=runoff_non_null, c=latitude_non_null,
                            s= np.interp(alt_non_null, (min(alt_non_null), max(alt_non_null)), (10, 100)),
                            alpha=.3) # t2m_non_null
            cbar1 = plt.colorbar(s1)
            cbar2 = plt.colorbar(s2)
            plt.show()

        #print(f"Valid measurements: R:{runoff_non_null.shape} T:{t2m_non_null.shape}") 30287,

        month_data = {
            'month': np.full(latitude_non_null.shape, month),
            'year': np.full(latitude_non_null.shape, year),
            'lat': latitude_non_null,
            'lon': longitude_non_null,
            'h': alt_non_null,
            't2m': t2m_non_null,
            'sw': swnetsrf_non_null,
            'runoff': runoff_non_null
        }
        # Append data to the DataFrame
        month_df = pd.DataFrame(month_data)

        if dataframe is None:
            dataframe = month_df  # Initialize the DataFrame at first iteration
        else:
            dataframe = pd.concat([dataframe, month_df], ignore_index=True)

        plot_t2m_run = False
        if plot_t2m_run:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ds_glc['RUNOFF'].plot(ax=ax1)
            ds_asm['T2M'].where(mask_non_null).plot(ax=ax2)
            plt.show()

    return dataframe

dataframe = build_dataset(year_in=2002, year_fin=2003)

dataframe = dataframe.sample(frac=1).reset_index(drop=True)


# Construct train and test datasets
dataframe = dataframe.loc[dataframe['t2m'] >= CFG.min_t2m]

X, y = dataframe[CFG.cols_train], dataframe[CFG.target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True)

print(f"Train: {X_train.shape} Test: {X_test.shape}")

model = CFG.model

model.fit(X_train, y_train)
y_preds = model.predict(X_test)

res = y_test - y_preds
print(f"Mean runoff: {np.mean(y_test)}, Mean res: {np.mean(res)}, stdev: {np.std(res)}, stdevperc: {100*np.std(res)/np.mean(y_test)} %")

m, q, r, p, se = stats.linregress(y_test, y_preds)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(x=y_test, y=y_preds)
ax1.axline(xy1=(min(y_test), m * min(y_test) + q), slope=m, c='r', lw=2, label=f"$y = {m:.2f}x + {q:.2e}$")
ax1.legend()
ax2.hist(res, bins=50, label='ML', color='lightblue', ec='blue', alpha=.4)
ax2.set_yscale('log')
plt.show()

#fig, (ax1, ax2) = plt.subplots(2,1)
#ax1.scatter(x=X_test['lon'], y=X_test['lat'], c=y_test, vmin=min(y_test), vmax=max(y_test), s=10)
#ax2.scatter(x=X_test['lon'], y=X_test['lat'], c=y_preds, vmin=min(y_test), vmax=max(y_test), s=10)
#plt.show()

# -- Model deploy
month, year = 1, 2010
for month in [1, 4, 9, 12]:
    file_asm = glob.glob(f"{args.merra2_input_folder}{year}/asm/MERRA2_*.instM_2d_asm_Nx.{year}{month:02}.nc4")[0]
    file_glc = glob.glob(f"{args.merra2_input_folder}{year}/glc/MERRA2_*.tavgM_2d_glc_Nx.{year}{month:02}.nc4")[0]
    file_int = glob.glob(f"{args.merra2_input_folder}{year}/int/MERRA2_*.tavgM_2d_int_Nx.{year}{month:02}.nc4")[0]
    ds_asm = rioxarray.open_rasterio(file_asm, crs="EPSG:4326")
    ds_glc = rioxarray.open_rasterio(file_glc, crs="EPSG:4326")
    ds_int = rioxarray.open_rasterio(file_int, crs="EPSG:4326")
    ds_glc['RUNOFF'] = ds_glc['RUNOFF'].where(ds_glc['RUNOFF'] != ds_glc['RUNOFF'].rio.nodata)
    ds_asm['T2M'] = ds_asm['T2M'].where(ds_asm['T2M'] != ds_asm['T2M'].rio.nodata)
    ds_int['SWNETSRF'] = ds_int['SWNETSRF'].where(ds_int['SWNETSRF'] != ds_int['SWNETSRF'].rio.nodata)

    lats = ds_asm['y'].values
    lons = ds_asm['x'].values
    h = merra_const['H'].values.squeeze()
    h_flat = h.flatten()
    t2m = ds_asm['T2M'].values.squeeze()
    t2m_flat = t2m.flatten()
    sw = ds_int['SWNETSRF'].values.squeeze()
    sw_flat = sw.flatten()

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lons_flat = lon_grid.flatten()
    lats_flat = lat_grid.flatten()
    months_flat = np.full(t2m_flat.shape, month)
    years_flat = np.full(t2m_flat.shape, year)

    data = {
            'month': months_flat,
            'year': years_flat,
            'lat': lats_flat,
            'lon': lons_flat,
            'h': h_flat,
            't2m': t2m_flat,
            'sw': sw_flat
        }
    X_deploy = pd.DataFrame(data)[CFG.cols_train]


    runoff_2d = model.predict(X_deploy).reshape(ds_asm['T2M'].values.shape)

    # Force runoff to 1e-9 for t2m<min_t2m and for negative predicted runoff
    runoff_2d[ds_asm['T2M'].values < CFG.min_t2m] = CFG.force_runoff_value
    runoff_2d[runoff_2d<CFG.min_runoff] = CFG.force_runoff_value

    ds_glc['RUNOFF_ML'] = (('time', 'y', 'x'), runoff_2d)

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    vmin, vmax = np.nanmin(runoff_2d), np.nanmax(runoff_2d)
    im1 = ds_glc['RUNOFF_ML'].plot(ax=ax1, norm=LogNorm(), vmin=vmin, vmax=vmax, add_colorbar=False)
    im2 =ds_glc['RUNOFF'].plot(ax=ax2, norm=LogNorm(), vmin=vmin, vmax=vmax, add_colorbar=False)
    s3 = ax3.scatter(x=ds_glc['RUNOFF'].values.flatten(), y=ds_glc['RUNOFF_ML'].values.flatten())
    utils.get_rgi_dir(version='62')  # setup oggm version
    for rgi in range(13, 16):
        oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
        gl_geoms = oggm_rgi_glaciers['geometry']
        gl_geoms_ext_gs = gl_geoms.exterior
        gl_geoms_ext_gdf = gpd.GeoDataFrame(geometry=gl_geoms_ext_gs, crs="EPSG:4326")
        gl_geoms_ext_gdf.plot(ax=ax1, linewidth=0.5, color='grey')
        gl_geoms_ext_gdf.plot(ax=ax2, linewidth=0.5, color='grey')
    
    cbar1 = plt.colorbar(im1, norm=LogNorm(), fraction=0.07, pad=0.04)
    cbar2 = plt.colorbar(im2, norm=LogNorm(), fraction=0.07, pad=0.04)
    plt.tight_layout()
    plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1,2)
    im1 = ds_glc['RUNOFF_ML'].where(~np.isnan(ds_glc['RUNOFF'])).plot(ax=ax1, cmap='binary')
    im2 = ds_glc['RUNOFF'].plot(ax=ax2, cmap='binary')
    plt.show()


input('Proceed to train_full_and_save...')
train_full_and_save = True
if train_full_and_save:
    dataframe = build_dataset(year_in=2002, year_fin=2020)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    dataframe = dataframe.loc[dataframe['t2m'] >= CFG.min_t2m]

    X, y = dataframe[CFG.cols_train], dataframe[CFG.target]

    print(f"Train with full data: {X.shape}")

    model = CFG.model

    model.fit(X, y)

    # -- Model deploy and save RUNOFF_ML in the same file
    for i, (year, month) in enumerate(product(range(2002, 2020), range(1, 13))):
        file_asm = glob.glob(f"{args.merra2_input_folder}{year}/asm/MERRA2_*.instM_2d_asm_Nx.{year}{month:02}.nc4")[0]
        file_glc = glob.glob(f"{args.merra2_input_folder}{year}/glc/MERRA2_*.tavgM_2d_glc_Nx.{year}{month:02}.nc4")[0]
        file_int = glob.glob(f"{args.merra2_input_folder}{year}/int/MERRA2_*.tavgM_2d_int_Nx.{year}{month:02}.nc4")[0]
        ds_asm = rioxarray.open_rasterio(file_asm, crs="EPSG:4326")
        ds_glc = rioxarray.open_rasterio(file_glc, crs="EPSG:4326")
        ds_int = rioxarray.open_rasterio(file_int, crs="EPSG:4326")

        list_datasets = [ds_int, ds_glc, ds_asm]
        for raster in list_datasets: raster.rio.write_crs("EPSG:4326", inplace=True)

        ds_glc['RUNOFF'] = ds_glc['RUNOFF'].where(ds_glc['RUNOFF'] != ds_glc['RUNOFF'].rio.nodata)
        ds_asm['T2M'] = ds_asm['T2M'].where(ds_asm['T2M'] != ds_asm['T2M'].rio.nodata)
        ds_int['SWNETSRF'] = ds_int['SWNETSRF'].where(ds_int['SWNETSRF'] != ds_int['SWNETSRF'].rio.nodata)

        lats = ds_asm['y'].values
        lons = ds_asm['x'].values
        h = merra_const['H'].values.squeeze()
        h_flat = h.flatten()
        t2m = ds_asm['T2M'].values.squeeze()
        t2m_flat = t2m.flatten()
        sw = ds_int['SWNETSRF'].values.squeeze()
        sw_flat = sw.flatten()

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lons_flat = lon_grid.flatten()
        lats_flat = lat_grid.flatten()
        months_flat = np.full(t2m_flat.shape, month)
        years_flat = np.full(t2m_flat.shape, year)

        data = {
            'month': months_flat,
            'year': years_flat,
            'lat': lats_flat,
            'lon': lons_flat,
            'h': h_flat,
            't2m': t2m_flat,
            'sw': sw_flat
        }
        X_deploy = pd.DataFrame(data)[CFG.cols_train]

        runoff_2d = model.predict(X_deploy).reshape(ds_asm['T2M'].values.shape)

        # Force runoff to 1e-9 for t2m<min_t2m and for negative predicted runoff
        runoff_2d[ds_asm['T2M'].values < CFG.min_t2m] = CFG.force_runoff_value
        runoff_2d[runoff_2d < CFG.min_runoff] = CFG.force_runoff_value

        ds_glc['RUNOFF_ML'] = (('time', 'y', 'x'), runoff_2d)

        #ds_glc['RUNOFF_ML'].plot(cmap='binary')
        #plt.show()

        ds_glc.rio.write_crs("EPSG:4326", inplace=True)

        # Save the dataset to a NetCDF file
        file_out = file_glc.replace('.nc4', '_ML.nc4')
        ds_glc.to_netcdf(file_out)
        print(f"Saved: {file_out}")