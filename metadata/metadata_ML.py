import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from glob import glob

from sklearn.preprocessing import QuantileTransformer

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb

from fetch_glacier_metadata import populate_glacier_with_metadata

#from umap import UMAP

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_final2_grid_20.csv",
                    help="Training dataset.")
parser.add_argument('--save_model', type=bool, default=False, help="True to save the model.")
parser.add_argument('--save_outdir', type=str, default="/home/nico/PycharmProjects/skynet/code/metadata/", help="Saved model dir.")
parser.add_argument('--save_outname', type=str, default="", help="Saved model name.")
args = parser.parse_args()

class CFG:

    features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
                'dist_from_border_km_geom', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
                'elevation_from_zmin', 'RGI']
    target = 'THICKNESS'
    millan = 'ith_m'
    farinotti = 'ith_f'

# Import the training dataset
glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)
glathida_rgis = glathida_rgis.loc[glathida_rgis['THICKNESS']>=0]
print(f"Dataset: {len(glathida_rgis)} rows, {glathida_rgis['RGI'].nunique()} regions and {glathida_rgis['RGIId'].nunique()} glaciers.")

def create_test(df, minimum_test_size=1000, rgi=None, seed=None):
    """
    - rgi se voglio creare il test in una particolare regione
    - minimum_test_size: quanto lo voglio grande
    """
    if rgi is not None:
        df = df[df['RGI']==rgi]
    if seed is not None:
        random.seed(seed)

    unique_glaciers = df['RGIId'].unique()
    random.shuffle(unique_glaciers)
    selected_glaciers = []
    n_total_points = 0

    for glacier_name in unique_glaciers:
        if n_total_points < minimum_test_size:
            selected_glaciers.append(glacier_name)
            n_points = df[df['RGIId'] == glacier_name].shape[0]
            n_total_points += n_points
            #print(glacier_name, n_points, n_total_points)
        else:
            #print('Finished with', n_total_points, 'points, and', len(selected_glaciers), 'glaciers.')
            break

    test = df[df['RGIId'].isin(selected_glaciers)]
    #print(test['RGI'].value_counts())
    #print('Total test size: ', len(test))
    return test

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y,predictions)
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R-SQUARED: {r_squared}')
    print(f'm: {slope}')
    print(f'q: {intercept}')
    print(f'r: {r_value}')
    return mae, mse, rmse, r_squared, slope, intercept, r_value


# Train, val, and test
test = create_test(glathida_rgis,  minimum_test_size=1800, rgi=7, seed=None)
val = glathida_rgis.drop(test.index).sample(n=1000)
train = glathida_rgis.drop(test.index).drop(val.index)

y_train, y_test = train[CFG.target], test[CFG.target]
X_train, X_test = train[CFG.features], test[CFG.features]

print(f'Dataset sizes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')

y_test_m = test[CFG.millan]#.to_numpy()
y_test_f = test[CFG.farinotti]#.to_numpy()

### LightGBM
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Plot feature importance
run_feature_importance = False
if run_feature_importance:
    lgb.plot_importance(model, importance_type="auto", figsize=(7,6), title="LightGBM Feature Importance")

#y_preds = model.predict(X_test.to_numpy())
y_preds = model.predict(X_test)

# benchmarks
ltb_metrics = evaluation(y_test, y_preds)

# stats
mu_ML = np.mean(y_test-y_preds)
med_ML = np.median(y_test-y_preds)
std_ML = np.std(y_test-y_preds)
mu_millan = np.mean(y_test-y_test_m)
med_millan = np.median(y_test-y_test_m)
std_millan = np.std(y_test-y_test_m)
mu_farinotti = np.mean(y_test-y_test_f)
med_farinotti = np.median(y_test-y_test_f)
std_farinotti = np.std(y_test-y_test_f)
print(f'Benchmarks ML, Millan and Farinotti: {std_ML:.2f} {std_millan:.2f} {std_farinotti:.2f}')

# fits
m_ML, q_ML, _, _, _ = stats.linregress(y_test, y_preds)
m_millan, q_millan, _, _, _ = stats.linregress(y_test, y_test_m)
m_farinotti, q_farinotti, _, _, _ = stats.linregress(y_test, y_test_f)

# plot
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
s1 = ax1.scatter(x=y_test, y=test['ith_m'], s=5, c='g', alpha=.3)
s1 = ax1.scatter(x=y_test, y=test['ith_f'], s=5, c='r', alpha=.3)
s1 = ax1.scatter(x=y_test, y=y_preds, s=5, c='aliceblue', ec='b', alpha=.5)

xmax = max(np.max(y_test), np.max(y_preds), np.max(y_test_m), np.max(y_test_f))

fit_ML_plot = ax1.plot([0.0, xmax], [q_ML, q_ML+xmax*m_ML], c='b')
fit_millan_plot = ax1.plot([0.0, xmax], [q_millan, q_millan+xmax*m_millan], c='lime')
fit_farinotti_plot = ax1.plot([0.0, xmax], [q_farinotti, q_farinotti+xmax*m_farinotti], c='r')
s2 = ax1.plot([0.0, xmax], [0.0, xmax], c='k')
ax1.axis([None, xmax, None, xmax])

ax2.hist(y_test-y_preds, bins=np.arange(-1000, 1000, 10), label='ML', color='lightblue', ec='blue', alpha=.4, zorder=2)
ax2.hist(y_test-y_test_m, bins=np.arange(-1000, 1000, 10), label='Millan', color='green', ec='green', alpha=.3, zorder=1)
ax2.hist(y_test-y_test_f, bins=np.arange(-1000, 1000, 10), label='Farinotti', color='red', ec='red', alpha=.3, zorder=1)

# text
text_ml = f'ML\n$\mu$ = {mu_ML:.1f}\nmed = {med_ML:.1f}\n$\sigma$ = {std_ML:.1f}'
text_millan = f'Millan\n$\mu$ = {mu_millan:.1f}\nmed = {med_millan:.1f}\n$\sigma$ = {std_millan:.1f}'
text_farinotti = f'Farinotti\n$\mu$ = {mu_farinotti:.1f}\nmed = {med_farinotti:.1f}\n$\sigma$ = {std_farinotti:.1f}'
# text boxes
props_ML = dict(boxstyle='round', facecolor='lightblue', ec='blue', alpha=0.4)
props_millan = dict(boxstyle='round', facecolor='lime', ec='green', alpha=0.4)
props_farinotti = dict(boxstyle='round', facecolor='salmon', ec='red', alpha=0.4)
ax2.text(0.05, 0.95, text_ml, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_ML)
ax2.text(0.05, 0.7, text_millan, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_millan)
ax2.text(0.05, 0.45, text_farinotti, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_farinotti)

ax1.set_xlabel('GT ice thickness (m)')
ax1.set_ylabel('Modelled ice thickness (m)')
ax2.set_xlabel('GT - Model (m)')
ax2.legend(loc='best')

plt.tight_layout()
plt.show()

# Visualize test predictions
print('Test dataset:', len(test))
test_glaciers_names = test['RGIId'].unique().tolist()
print('Test glaciers:', test_glaciers_names)

glacier_geometries = []
for glacier_name in test_glaciers_names:
    rgi = glacier_name[6:8]
    oggm_rgi_shp = glob(f'/home/nico/OGGM/rgi/RGIV62/{rgi}*/{rgi}*.shp')[0]
    print(glacier_name)
    oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
    glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name]['geometry'].item()
    #print(glacier_geometry)
    glacier_geometries.append(glacier_geometry)


fig, axes = plt.subplots(2,3, figsize=(10,7))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

y_min = min(np.concatenate((y_test, y_preds, y_test_m, y_test_f)))
y_max = max(np.concatenate((y_test, y_preds, y_test_m, y_test_f)))
y_min_diff = min(np.concatenate((y_preds-y_test_f, y_test-y_preds)))
y_max_diff = max(np.concatenate((y_preds-y_test_f, y_test-y_preds)))
absmax = max(abs(y_min_diff), abs(y_max_diff))

s1 = ax1.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test, cmap='Blues', label='GT')
s2 = ax2.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_preds, cmap='Blues', label='ML')
s3 = ax3.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test_m, cmap='Blues', label='Millan')
s4 = ax4.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test_f, cmap='Blues', label='Farinotti')
s5 = ax5.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=(y_test-y_preds)/y_test, cmap='bwr', label='GT-ML')
s6 = ax6.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=(y_test-y_test_f)/y_test, cmap='bwr', label='GT-Farinotti')

for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
    for geom in glacier_geometries:
        ax.plot(*geom.exterior.xy, c='k')

cbar1 = plt.colorbar(s1, ax=ax1)
cbar2 = plt.colorbar(s2, ax=ax2)
cbar3 = plt.colorbar(s3, ax=ax3)
cbar4 = plt.colorbar(s4, ax=ax4)
cbar5 = plt.colorbar(s5, ax=ax5)
cbar6 = plt.colorbar(s6, ax=ax6)

for cbar in (cbar1, cbar2, cbar3, cbar4):
    cbar.mappable.set_clim(vmin=y_min,vmax=y_max)
    cbar.set_label('THICKNESS (m)', labelpad=15, rotation=270)
for cbar in (cbar5, cbar6):
    cbar.mappable.set_clim(vmin=-3, vmax=3)

for ax in (ax1, ax2, ax3, ax4, ax5, ax6): ax.legend(loc='upper left')

plt.show()

# Visualize test predictions of specific glacier
# Generate points for one glacier
glacier_name_for_generation = 'RGI60-07.00228' #RGI60–07.00027 'RGI60-11.01450'
test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation, n=2000)

X_train_glacier = test_glacier[CFG.features]
y_test_glacier_m = test_glacier[CFG.millan]
y_test_glacier_f = test_glacier[CFG.farinotti]

y_preds_glacier = model.predict(X_train_glacier)

rgi = glacier_name_for_generation[6:8]
oggm_rgi_shp = glob(f'/home/nico/OGGM/rgi/RGIV62/{rgi}*/{rgi}*.shp')[0]
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['geometry'].item()

y_min = min(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))
y_max = max(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))

fig, axes = plt.subplots(1,3, figsize=(5,3))
ax1, ax2, ax3 = axes.flatten()
s1 = ax1.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=10, c=y_preds_glacier, cmap='Blues', label='ML')
s2 = ax2.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=10, c=y_test_glacier_m, cmap='Blues', label='Millan')
s3 = ax3.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=10, c=y_test_glacier_f, cmap='Blues', label='Farinotti')

cbar1 = plt.colorbar(s1, ax=ax1)
cbar2 = plt.colorbar(s2, ax=ax2)
cbar3 = plt.colorbar(s3, ax=ax3)

for cbar in (cbar1, cbar2, cbar3, cbar4):
    cbar.mappable.set_clim(vmin=y_min,vmax=y_max)
    cbar.set_label('THICKNESS (m)', labelpad=15, rotation=270)

for ax in (ax1, ax2, ax3):
    ax.plot(*glacier_geometry.exterior.xy, c='k')

for ax in (ax1, ax2, ax3):
    ax.legend(loc='upper left')
    #ax.set_xlabel('')
    #ax.set_ylabel('')
    ax.axis("off")

fig.suptitle(f'{glacier_name_for_generation}', fontsize=13)
plt.tight_layout()
plt.show()