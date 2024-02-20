import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from glob import glob
import rioxarray
from oggm import utils

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
import optuna

from fetch_glacier_metadata import populate_glacier_with_metadata

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/"
                        +"glathida-3.1.0/data/metadata_hmineq0.0_tmin20050000_mean_grid_20.csv", help="Training dataset.")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/home/nico/PycharmProjects/skynet/Extra_Data/Farinotti/",
                    help="Path to Farinotti ice thickness data")
parser.add_argument('--save_model', type=bool, default=False, help="True to save the model.")
parser.add_argument('--save_outdir', type=str, default="/home/nico/PycharmProjects/skynet/code/metadata/", help="Saved model dir.")
parser.add_argument('--save_outname', type=str, default="", help="Saved model name.")
args = parser.parse_args()

utils.get_rgi_dir(version='62')  # setup oggm version
utils.get_rgi_intersects_dir(version='62')

"""
#pd.set_option('display.max_columns', 500)
#pd.set_option('display.max_rows', 1615)
oggm_rgi_shp = "/home/nico/OGGM/rgi/RGIV62/08_rgi62_Scandinavia/08_rgi62_Scandinavia.shp"
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp).sort_values(by=['Area'], ascending=False)
#print(oggm_rgi_glaciers['GLIMSId'])
#print(oggm_rgi_glaciers.head(10))
print(oggm_rgi_glaciers.loc[:30, ['Name', 'Area', 'RGIId']])
#oggm_rgi_glaciers2 = gpd.read_file(oggm_rgi_shp).sort_values(by=['GLIMSId'])
#print(oggm_rgi_glaciers2.loc[816, ['Area', 'Name', 'GLIMSId']])
#print(oggm_rgi_glaciers[oggm_rgi_glaciers['GLIMSId'] == 'G013542E78988N'])"""


class CFG:
    features_not_used = ['RGI', 'dvx', 'dvy', ]
    features = ['Area', 'slope_lon_gf50', 'slope_lat_gf50', 'elevation_astergdem', 'vx_gf300', 'vy_gf300',
                'dist_from_border_km_geom',  'slope50','slope150','slope300', 'slope450', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
                'elevation_from_zmin', 'Form', 'TermType', 'Aspect', 'curv_50', 'curv_300', 'aspect_50', 'aspect_300',
                'dvx_dx', 'dvx_dy', 'dvy_dx','dvy_dy',  'v50', 'v150', 'v300',]
    target = 'THICKNESS'
    millan = 'ith_m'
    farinotti = 'ith_f'
    model = lgb.LGBMRegressor(num_leaves=28, n_jobs=12)
    #model = xgb.XGBRegressor()
    # model = RandomForestRegressor()
    n_rounds = 500
    use_log_transform = False
    run_feature_importance = False


# Import the training dataset
glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)
# Filter out some portion of it
glathida_rgis = glathida_rgis.loc[glathida_rgis['THICKNESS']>=0]

# Add some features
glathida_rgis['v50'] = np.sqrt(glathida_rgis['vx_gf50']**2 + glathida_rgis['vy_gf50']**2)
glathida_rgis['v150'] = np.sqrt(glathida_rgis['vx_gf150']**2 + glathida_rgis['vy_gf150']**2)
glathida_rgis['v300'] = np.sqrt(glathida_rgis['vx_gf300']**2 + glathida_rgis['vy_gf300']**2)
glathida_rgis['dvx'] = np.sqrt(glathida_rgis['dvx_dx']**2 + glathida_rgis['dvx_dy']**2) #todo: wrong ! df = dfx/dx dx + dfy/dy dy
glathida_rgis['dvy'] = np.sqrt(glathida_rgis['dvy_dx']**2 + glathida_rgis['dvy_dy']**2) #todo: wrong !
glathida_rgis['slope50'] = np.sqrt(glathida_rgis['slope_lon_gf50']**2 + glathida_rgis['slope_lat_gf50']**2)
glathida_rgis['slope150'] = np.sqrt(glathida_rgis['slope_lon_gf150']**2 + glathida_rgis['slope_lat_gf150']**2)
glathida_rgis['slope300'] = np.sqrt(glathida_rgis['slope_lon_gf300']**2 + glathida_rgis['slope_lat_gf300']**2)
glathida_rgis['slope450'] = np.sqrt(glathida_rgis['slope_lon_gf450']**2 + glathida_rgis['slope_lat_gf450']**2)
glathida_rgis['elevation_from_zmin'] = glathida_rgis['elevation_astergdem'] - glathida_rgis['Zmin']
#glathida_rgis['hbahrm'] = 0.03*(glathida_rgis['Area']**0.375)*1000 # Bahr's approximation: h in meters
#glathida_rgis['hbahrm2'] = glathida_rgis['dist_from_border_km_geom']*np.sqrt(glathida_rgis['Area'])
A = 24 * np.power(10, -25.0) #s−1 Pa−3
rho, g = 917., 9.81
#glathida_rgis['sia'] = ((.5*glathida_rgis['v']*(3+1))/(2*A*(rho*g*glathida_rgis['v'])**3))**(1./4)

print(f"Overall dataset: {len(glathida_rgis)} rows, {glathida_rgis['RGI'].nunique()} regions and {glathida_rgis['RGIId'].nunique()} glaciers.")

# umap and tsne
run_umap_tsne = False
if run_umap_tsne:
    print(f"Begin umap and tsne")
    import umap
    from sklearn.manifold import TSNE

    reducer = umap.UMAP(n_neighbors=5, min_dist=0.05, n_components=2, metric='euclidean')
    embedding_umap = reducer.fit_transform(glathida_rgis[CFG.features])
    embeddeding_tsne = TSNE(n_components=2).fit_transform(glathida_rgis[CFG.features])
    print(embedding_umap.shape)
    print(embeddeding_tsne.shape)

    fig, (ax1, ax2) = plt.subplots(1,2)
    s1 = ax1.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=glathida_rgis[CFG.target], cmap='gnuplot', s=5)
    cbar1 = plt.colorbar(s1, ax=ax1, alpha=1)
    cbar1.set_label('THICKNESS (m)', labelpad=15, rotation=270)
    s2 = ax2.scatter(embeddeding_tsne[:, 0], embeddeding_tsne[:, 1], c=glathida_rgis[CFG.target], cmap='gnuplot', s=5)
    cbar2 = plt.colorbar(s2, ax=ax2, alpha=1)
    cbar2.set_label('THICKNESS (m)', labelpad=15, rotation=270)
    plt.show()
    input('wait')


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

def compute_scores(y, predictions, verbose=False):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y,predictions)
    res = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-SQUARED': r_squared, 'm': slope, 'q': intercept, 'r': r_value}
    if verbose:
        for key in res: print(f"{key}: {res[key]:.2f}", end=", ")
    return res

def objective(trial):

    # Suggest values of the hyperparameters using a trial object.
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 30),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    test = create_test(glathida_rgis, minimum_test_size=1800, rgi=None, seed=42)
    train = glathida_rgis.drop(test.index)
    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_preds, squared=False)
    return rmse

optune_optimize = False
if optune_optimize:
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print('Best hyperparameters:', study.best_params)
    print('Best RMSE:', study.best_value)

    input('Continue')

stds_ML, meds_ML, slopes_ML = [], [], []
stds_Mil, meds_Mil, slopes_Mil = [], [], []
stds_Far, meds_Far, slopes_Far = [], [], []

best_model = None
best_slope = -999
best_rmse = 9999

for i in range(CFG.n_rounds):

    # Train, val, and test
    test = create_test(glathida_rgis,  minimum_test_size=round(0.06*len(glathida_rgis)), rgi=None, seed=None)

    create_val = False
    if create_val:
        val = glathida_rgis.drop(test.index).sample(n=500)
        train = glathida_rgis.drop(test.index).drop(val.index)
    else: train = glathida_rgis.drop(test.index)

    print(f"Iter {i} Train/Test: {len(train)}/{len(test)}")

    plot_some_train_features = False
    if plot_some_train_features:
        fig, ax = plt.subplots()
        ax.hist(train['slope_lon_gf300'], bins=np.arange(train['slope_lon_gf300'].min(), train['slope_lon_gf300'].max(), 0.1), color='k', ec='k', alpha=.4)
        plt.show()

    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]
    # print(f'Dataset sizes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')
    y_test_m = test[CFG.millan]
    y_test_f = test[CFG.farinotti]

    # Log transform the target variable
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    ### LightGBM
    model = CFG.model

    if CFG.use_log_transform:
        model.fit(X_train, y_train_log)
        y_preds_log = model.predict(X_test)
        y_preds = np.expm1(y_preds_log)

    else:
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

    # Run feature importance
    if CFG.run_feature_importance:
        lgb.plot_importance(model, importance_type="auto", figsize=(7,6), title="LightGBM Feature Importance")
        plt.show()

    # benchmarks
    model_metrics = compute_scores(y_test, y_preds, verbose=False)

    #*** Note: here it is very important since this is the policy to decide which model will be selected for deploy
    m = model_metrics['m']
    rmse = model_metrics['RMSE']
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

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
    print(f'{i} Benchmarks ML, Millan and Farinotti: {std_ML:.2f} {std_millan:.2f} {std_farinotti:.2f}')

    # fits
    m_ML, q_ML, _, _, _ = stats.linregress(y_test, y_preds)
    m_millan, q_millan, _, _, _ = stats.linregress(y_test, y_test_m)
    m_farinotti, q_farinotti, _, _, _ = stats.linregress(y_test, y_test_f)

    stds_ML.append(std_ML)
    meds_ML.append(med_ML)
    slopes_ML.append(m_ML)
    stds_Mil.append(std_millan)
    meds_Mil.append(med_millan)
    slopes_Mil.append(m_millan)
    stds_Far.append(std_farinotti)
    meds_Far.append(med_farinotti)
    slopes_Far.append(m_farinotti)

print(f"Res. medians {np.mean(meds_ML):.2f}({np.std(meds_ML):.2f}) {np.mean(meds_Mil):.2f}({np.std(meds_Mil):.2f}) {np.mean(meds_Far):.2f}({np.std(meds_Far):.2f})")
print(f"Res. stdevs {np.mean(stds_ML):.2f}({np.std(stds_ML):.2f}) {np.mean(stds_Mil):.2f}({np.std(stds_Mil):.2f}) {np.mean(stds_Far):.2f}({np.std(stds_Far):.2f})")
print(f"Res. slopes {np.mean(slopes_ML):.2f}({np.std(slopes_ML):.2f}) {np.mean(slopes_Mil):.2f}({np.std(slopes_Mil):.2f}) {np.mean(slopes_Far):.2f}({np.std(slopes_Far):.2f})")

print(f"At the end of cv the best rmse is {best_rmse}")

# ************************************
# plot
# ************************************
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

ax2.hist(y_test-y_preds, bins=np.arange(-xmax, xmax, 10), label='ML', color='lightblue', ec='blue', alpha=.4, zorder=2)
ax2.hist(y_test-y_test_m, bins=np.arange(-xmax, xmax, 10), label='Millan', color='green', ec='green', alpha=.3, zorder=1)
ax2.hist(y_test-y_test_f, bins=np.arange(-xmax, xmax, 10), label='Farinotti', color='red', ec='red', alpha=.3, zorder=1)

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

plot_spatial_test_predictions = False
if plot_spatial_test_predictions:

    # Visualize test predictions
    test_glaciers_names = test['RGIId'].unique().tolist()
    print(f"Test dataset: {len(test)} points and {len(test_glaciers_names)} glaciers")

    glacier_geometries = []
    for glacier_name in test_glaciers_names:
        rgi = glacier_name[6:8]
        oggm_rgi_shp = glob(f'/home/nico/OGGM/rgi/RGIV62/{rgi}*/{rgi}*.shp')[0]
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
        glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == glacier_name]['geometry'].item()
        # print(glacier_geometry)
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

# *********************************************
# Model deploy
# *********************************************
def get_random_glacier_rgiid(name=None, rgi=11, area=None, seed=None):
    """Provide a rgi number and seed. This method returns a
    random glacier rgiid name.
    If not rgi is passed, any rgi region is good.
    """
    if name is not None: return name
    if seed is not None:
        np.random.seed(seed)
    if rgi is not None:
        oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')
        oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
    if area is not None:
        oggm_rgi_glaciers = oggm_rgi_glaciers[oggm_rgi_glaciers['Area'] > area]
    rgi_ids = oggm_rgi_glaciers['RGIId'].dropna().unique().tolist()
    rgiid = np.random.choice(rgi_ids)
    return rgiid

glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-11.01450', rgi=8, area=None, seed=None)
#glacier_name_for_generation = 'RGI60-07.00228' #RGI60–07.00027 'RGI60-11.01450' RGI60-07.00552,'RGI60-07.00228'
#glacier_name_for_generation = 'RGI60-07.00832' very nice
#'RGI60-03.01632', 'RGI60-07.01482' ML simile agli altri 2 in termini di alte profondita
# 'RGI60-03.00251' Dobbin Bay, 'RGI60-07.00124' Renardbreen, 'RGI60-11.01328' Unteraargletscher, 'RGI60-11.01478'
# 'RGI60-03.02469', 'RGI60-03.04338', 'RGI60-03.01483 ML << Millan/Farinotti
# 'RGI60-03.00228'
# 'RGI60-11.01492' we can see millan's effect of velocity products on ice thickness calculation
# RGI60-07.00027 biggest in Svalbard
# RGI60-03.01710 biggest in Arctic Canada (Wykeham Glacier South)
# 'RGI60-07.01464' Holtedahlfonna
# in 'RGI60-08.01657' and RGI60-08.01641 I see Millan having gaps (in v hence in ith_m).
# no Millan data: RGI60-08.03159, RGI60-08.03084 controlla questo
# 'RGI60-03.01062' is so small that has negative predictions !
# RGI60-03.00862 has issues with projections,
# fix bug for 'RGI60-03.04229' !!!!!
# RGI60-03.02811 in interesting since on top Millan has no data, so what is the effect or modeling with/without v ?

try:
    test_glacier_rgi = glacier_name_for_generation[6:8]
    test_glacier_folder_farinotti = glob(f"{args.farinotti_icethickness_folder}/*{test_glacier_rgi}/*{test_glacier_rgi}/")[0]
    ice_farinotti = rioxarray.open_rasterio(test_glacier_folder_farinotti+glacier_name_for_generation+'_thickness.tif')
    res_farinotti = ice_farinotti.rio.resolution()[0]
    vol_farinotti = 1.e-9 * (res_farinotti ** 2) * np.nansum(ice_farinotti.values) # Volume Farinotti km3
except ValueError:
    print(f"Farinotti glacier {glacier_name_for_generation} not found.")


# Generate points for one glacier
test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation, n=10000, seed=42)
test_glacier['v50'] = np.sqrt(test_glacier['vx_gf50']**2 + test_glacier['vy_gf50']**2)
test_glacier['v150'] = np.sqrt(test_glacier['vx_gf150']**2 + test_glacier['vy_gf150']**2)
test_glacier['v300'] = np.sqrt(test_glacier['vx_gf300']**2 + test_glacier['vy_gf300']**2)
test_glacier['dvx'] = np.sqrt(test_glacier['dvx_dx']**2 + test_glacier['dvx_dy']**2)
test_glacier['dvy'] = np.sqrt(test_glacier['dvy_dx']**2 + test_glacier['dvy_dy']**2)
test_glacier['slope50'] = np.sqrt(test_glacier['slope_lon_gf50']**2 + test_glacier['slope_lat_gf50']**2)
test_glacier['slope150'] = np.sqrt(test_glacier['slope_lon_gf150']**2 + test_glacier['slope_lat_gf150']**2)
test_glacier['slope300'] = np.sqrt(test_glacier['slope_lon_gf300']**2 + test_glacier['slope_lat_gf300']**2)
test_glacier['slope450'] = np.sqrt(test_glacier['slope_lon_gf450']**2 + test_glacier['slope_lat_gf450']**2)
test_glacier['elevation_from_zmin'] = test_glacier['elevation_astergdem'] - test_glacier['Zmin']
#test_glacier['sia'] = ((0.3*test_glacier['v']*(3+1))/(2*A*(rho*g*test_glacier['v'])**3))**(1./4)

X_train_glacier = test_glacier[CFG.features]
y_test_glacier_m = test_glacier[CFG.millan]  # Note that here nans are present if Millan has no data
no_millan_data = np.isnan(y_test_glacier_m).all()
y_test_glacier_f = test_glacier[CFG.farinotti]

if CFG.use_log_transform:
    y_preds_glacier_log = best_model.predict(X_train_glacier)
    y_preds_glacier = np.expm1(y_preds_glacier_log)  # Inverse log transform
else:
    y_preds_glacier = best_model.predict(X_train_glacier)

# Set negative predictions to zero
y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

rgi = glacier_name_for_generation[6:8]
oggm_rgi_shp = glob(f'/home/nico/OGGM/rgi/RGIV62/{rgi}*/{rgi}*.shp')[0]
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['geometry'].item()
glacier_area = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==glacier_name_for_generation]['Area'].item()
exterior_ring = glacier_geometry.exterior  # shapely.geometry.polygon.LinearRing
glacier_nunataks_list = [nunatak for nunatak in glacier_geometry.interiors]

def calc_volume_glacier(points_thickness, area=0):
    # A potential drawback of this method is that I am randomly sampling in epsg:4326. In a utm projection
    # such sampling does not turn out to be uniform. Returned volume in km3.
    N = len(points_thickness)
    volume = np.sum(points_thickness) * 0.001 * area / N
    return volume

vol_ML = calc_volume_glacier(y_preds_glacier, glacier_area)
print(f"Glacier {glacier_name_for_generation} Area: {glacier_area:.2f} km2, volML: {vol_ML:.3f} km3 volFar: {vol_farinotti:.3f} km3")
print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

# Visualize test predictions of specific glacier
y_min = min(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))
y_max = max(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))

fig, axes = plt.subplots(1,3, figsize=(8,4))
ax1, ax2, ax3 = axes.flatten()
s1 = ax1.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_preds_glacier, cmap='Blues', label='ML')
#cntr1 = ax1.tricontourf(test_glacier['lons'], test_glacier['lats'], y_preds_glacier, levels=5, cmap="Blues")
if not no_millan_data:
    s2 = ax2.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_m, cmap='Blues', label='Millan')
#cntr2 = ax2.tricontourf(test_glacier['lons'], test_glacier['lats'], y_test_glacier_m, levels=5, cmap="Blues")
s3 = ax3.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_f, cmap='Blues', label='Farinotti')
#cntr3 = ax3.tricontourf(test_glacier['lons'], test_glacier['lats'], y_test_glacier_f, levels=5, cmap="Blues")
#im4 = ice_farinotti.plot(ax=ax4, cmap='Blues', vmin=y_min,vmax=y_max)

ax1.set_title(f"ML {vol_ML:.3f} km3")
ax2.set_title(f"Millan")
ax3.set_title(f"Farinotti {vol_farinotti:.3f} km3")

cbar1 = plt.colorbar(s1, ax=ax1)
cbar1.mappable.set_clim(vmin=y_min,vmax=y_max)
cbar1.set_label('THICKNESS (m)', labelpad=15, rotation=270)
if not no_millan_data:
    cbar2 = plt.colorbar(s2, ax=ax2)
    cbar2.mappable.set_clim(vmin=y_min, vmax=y_max)
    cbar2.set_label('THICKNESS (m)', labelpad=15, rotation=270)
cbar3 = plt.colorbar(s3, ax=ax3)
cbar3.mappable.set_clim(vmin=y_min,vmax=y_max)
cbar3.set_label('THICKNESS (m)', labelpad=15, rotation=270)

for ax in (ax1, ax2, ax3):
    ax.plot(*exterior_ring.xy, c='k')
    for nunatak in glacier_nunataks_list:
        ax.plot(*nunatak.xy, c='k', lw=0.8)

    #ax.legend(fontsize=14, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    #ax.axis("off")
#ax4.set_title("")
#ax4.axis("off")

fig.suptitle(f'{glacier_name_for_generation}', fontsize=13)
plt.tight_layout()
plt.show()