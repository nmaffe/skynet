import argparse, time
import random
from tqdm import tqdm
import copy, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import pandas as pd
import earthpy.spatial
import geopandas as gpd
from glob import glob
import xarray, rioxarray
from oggm import utils

from scipy import stats
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
import optuna
import shap
from fetch_glacier_metadata import populate_glacier_with_metadata
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import calc_volume_glacier, get_random_glacier_rgiid, create_train_test

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/media/maffe/nvme/glathida/glathida-3.1.0/"
                        +"glathida-3.1.0/data/metadata35_hmineq0.0_tmin20050000_mean_grid_100.csv", help="Training dataset.")
parser.add_argument('--mosaic', type=str,default="/media/maffe/nvme/Tandem-X-EDEM/", help="Path to Tandem-X-EDEM")
parser.add_argument('--oggm', type=str,default="/home/maffe/OGGM/", help="Path to OGGM folder")
parser.add_argument('--load_outdir', type=str, default="/home/maffe/PycharmProjects/skynet/metadata/", help="Saved model dir.")
args = parser.parse_args()

class CFG:
    featuresSmall = ['Area', 'Perimeter', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'Aspect', 'Form', 'TermType',
                     'elevation', 'elevation_from_zmin', 'dist_from_border_km_geom',
                     'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
                     'curv_50', 'curv_300', 'curv_gfa', 'dmdtda_hugo',
                     'smb', 't2m', 'dist_from_ocean']
    featuresBig = featuresSmall + ['v50', 'v100', 'v150', 'v300', 'v450', 'vgfa', ]
    target = 'THICKNESS'
    millan = 'ith_m'
    farinotti = 'ith_f'
    features = featuresBig

    n_points_regression = 30000

glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)
glathida_rgis = glathida_rgis[~glathida_rgis['RGIId'].isin(['RGI60-19.01406'])]
glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, 'THICKNESS'] = glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, ['ith_m', 'ith_f']].mean(axis=1, skipna=True)

# Load the model
model_filename = args.load_outdir + 'iceboost_20240711.json'
iceboost = xgb.Booster()
iceboost.load_model(model_filename)

# *********************************************
# Model deploy
# *********************************************
glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-04.03263', rgi=4, area=30, seed=None)

test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
                                              n=CFG.n_points_regression, seed=42, verbose=True)
test_glacier_rgi = glacier_name_for_generation[6:8]

# Begin to extract all necessary things to plot the result
oggm_rgi_shp = glob(f"{args.oggm}rgi/RGIV62/{test_glacier_rgi}*/{test_glacier_rgi}*.shp")[0]
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['geometry'].item()
glacier_area = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['Area'].item()
exterior_ring = glacier_geometry.exterior  # shapely.geometry.polygon.LinearRing
glacier_nunataks_list = [nunatak for nunatak in glacier_geometry.interiors]

swlat = test_glacier['lats'].min()
swlon = test_glacier['lons'].min()
nelat = test_glacier['lats'].max()
nelon = test_glacier['lons'].max()
deltalat = np.abs(swlat - nelat)
deltalon = np.abs(swlon - nelon)
eps = 5./3600
focus_mosaic_tiles = create_glacier_tile_dem_mosaic(minx=swlon - (deltalon + eps),
                            miny=swlat - (deltalat + eps),
                            maxx=nelon + (deltalon + eps),
                            maxy=nelat + (deltalat + eps),
                             rgi=test_glacier_rgi, path_tandemx=args.mosaic)
focus = focus_mosaic_tiles.squeeze()

X_test_glacier = test_glacier[CFG.features]
y_test_glacier_m = test_glacier[CFG.millan]
y_test_glacier_f = test_glacier[CFG.farinotti]

no_millan_data = np.isnan(y_test_glacier_m).all()
no_farinotti_data = np.isnan(y_test_glacier_f).all()

dtest = xgb.DMatrix(data=X_test_glacier)

y_preds_glacier = iceboost.predict(dtest)

# Calculate the glacier volume using the 3 models
vol_ML = calc_volume_glacier(y_preds_glacier, glacier_area)
vol_millan = calc_volume_glacier(y_test_glacier_m, glacier_area)
vol_farinotti = calc_volume_glacier(y_test_glacier_f, glacier_area)
print(f"Glacier {glacier_name_for_generation} Area: {glacier_area:.2f} km2, "
      f"volML: {vol_ML:.4g} km3 "
      f"volMil: {vol_millan:.4g} km3 "
      f"volFar: {vol_farinotti:.4g} km3")

print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

vmin = min(y_preds_glacier)
vmax = max(y_preds_glacier)

plot_fancy_ML_prediction = True
if plot_fancy_ML_prediction:
    fig, ax = plt.subplots(figsize=(8,6))

    x0, y0, x1, y1 = exterior_ring.bounds
    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
    hillshade = hillshade.rio.clip_box(minx=x0-dx/4, miny=y0-dy/4, maxx=x1+dx/4, maxy=y1+dy/4)

    im = hillshade.plot(ax=ax, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

    s1 = ax.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=y_preds_glacier,
                     cmap='jet', label='ML', zorder=1, vmin=vmin,vmax=vmax)
    s_glathida = ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                            cmap='jet', ec='grey', lw=0.5, s=35, vmin=vmin,vmax=vmax)

    cbar = plt.colorbar(s1, ax=ax)
    cbar.mappable.set_clim(vmin=vmin,vmax=vmax)
    cbar.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)

    cbar.ax.tick_params(labelsize=12)
    ax.plot(*exterior_ring.xy, c='k')
    for nunatak in glacier_nunataks_list:
        ax.plot(*nunatak.xy, c='k', lw=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=16)
    ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=16)
    ax.set_title('')

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    #plt.savefig('/home/maffe/Downloads/RGI60-1313574_CCAI.png', dpi=200)
    plt.show()


plot_fancy_ML_Mil_Far_prediction = True
if plot_fancy_ML_Mil_Far_prediction:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15, 6))

    x0, y0, x1, y1 = exterior_ring.bounds
    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
    hillshade = hillshade.rio.clip_box(minx=x0 - dx / 8, miny=y0 - dy / 8, maxx=x1 + dx / 8, maxy=y1 + dy / 8)

    im1 = hillshade.plot(ax=ax1, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
    im2 = hillshade.plot(ax=ax2, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
    im3 = hillshade.plot(ax=ax3, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)

    s1 = ax1.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_preds_glacier, cmap='jet', label='ML', vmin=vmin, vmax=vmax)
    if not no_millan_data:
        s2 = ax2.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_m, cmap='jet',
                         label='Millan', vmin=vmin, vmax=vmax)
    if not no_farinotti_data:
        s3 = ax3.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_f, cmap='jet',
                         label='Farinotti', vmin=vmin, vmax=vmax)

    ax1.set_title(f"IceBoost: {vol_ML:.4g} km$^3$", fontsize=16)
    ax2.set_title(f"Model1: {vol_millan:.4g} km$^3$", fontsize=16)
    ax3.set_title(f"Model2: {vol_farinotti:.4g} km$^3$", fontsize=16)

    for ax in (ax1, ax2, ax3):
        ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                                cmap='jet', ec='grey', lw=0.5, s=35, vmin=vmin, vmax=vmax)

    cbar1 = plt.colorbar(s1, ax=ax1)
    cbar1.mappable.set_clim(vmin=vmin, vmax=vmax)
    cbar1.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
    cbar1.ax.tick_params(labelsize=11)
    if not no_millan_data:
        cbar2 = plt.colorbar(s2, ax=ax2)
        cbar2.mappable.set_clim(vmin=vmin, vmax=vmax)
        cbar2.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
        cbar2.ax.tick_params(labelsize=11)
    if not no_farinotti_data:
        cbar3 = plt.colorbar(s3, ax=ax3)
        cbar3.mappable.set_clim(vmin=vmin, vmax=vmax)
        cbar3.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=16)
        cbar3.ax.tick_params(labelsize=11)

    for ax in (ax1, ax2, ax3):
        ax.plot(*exterior_ring.xy, c='k')
        for nunatak in glacier_nunataks_list:
            ax.plot(*nunatak.xy, c='k', lw=0.8)

        # ax.legend(fontsize=14, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('Lon ($^{\\circ}$E)', fontsize=14)
        ax.set_ylabel('Lat ($^{\\circ}$N)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        #ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax2.axis('off')
    ax3.axis('off')

    #fig.suptitle(f'{glacier_name_for_generation}', fontsize=13)
    plt.tight_layout()
    plt.show()