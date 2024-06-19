import argparse, time
import random
from tqdm import tqdm
import copy, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import earthpy.spatial
#import earthpy.plot
import geopandas as gpd
from glob import glob
import xarray, rioxarray
from oggm import utils

from scipy import stats
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE

import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
from fetch_glacier_metadata import populate_glacier_with_metadata
from create_rgi_mosaic_tanxedem import create_glacier_tile_dem_mosaic
from utils_metadata import calc_volume_glacier

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/media/maffe/nvme/glathida/glathida-3.1.0/"
                        +"glathida-3.1.0/data/metadata28_hmineq0.0_tmin20050000_mean_grid_100.csv", help="Training dataset.")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/media/maffe/nvme/Farinotti/composite_thickness_RGI60-all_regions/",
                    help="Path to Farinotti ice thickness data")
parser.add_argument('--mosaic', type=str,default="/media/maffe/nvme/Tandem-X-EDEM/", help="Path to Tandem-X-EDEM")
parser.add_argument('--oggm', type=str,default="/home/maffe/OGGM/", help="Path to OGGM folder")
parser.add_argument('--save_model', type=int, default=0, help="Save ave the model: 0/1")
parser.add_argument('--save_outdir', type=str, default="/home/maffe/PycharmProjects/skynet/metadata/", help="Saved model dir.")
parser.add_argument('--save_outname', type=str, default="", help="Saved model name.")

args = parser.parse_args()

# setup oggm version
utils.get_rgi_dir(version='62')
utils.get_rgi_intersects_dir(version='62')

class CFG:
    features_not_used = ['sia']

    featuresSmall = ['RGI', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'Form', 'TermType', 'Aspect',
                'elevation', 'elevation_from_zmin', 'dist_from_border_km_geom',
                  'slope50', 'slope75', 'slope100', 'slope125', 'slope150', 'slope300', 'slope450', 'slopegfa',
                'curv_50', 'curv_300', 'curv_gfa', 'aspect_50', 'aspect_300', 'aspect_gfa', 'lats', 'dmdtda_hugo', 'smb',
                     ]

    featuresBig = featuresSmall + ['v50', 'v100', 'v150', 'v300', 'v450', 'vgfa', ]

    target = 'THICKNESS'
    millan = 'ith_m'
    farinotti = 'ith_f'
    #model = lgb.LGBMRegressor(num_leaves=40, n_jobs=-1, loss='l2', verbose=-1) #num_leaves=28, 40
    xgb_loss = 'reg:squarederror' #'reg:squarederror' #'reg:absoluteerror'
    xgb_params = {'lambda': 0.0832200463578115,
                    'alpha': 6.296986987802592,
                    'colsample_bytree': 0.6995472160,
                    'subsample': 0.7687278029948124,
                    'learning_rate': 0.0323799185617,
                    'n_estimators': 537,
                    'max_depth': 15,
                    'min_child_weight': 10,
                    'gamma': 0.0803458919901354,}
    #model = xgb.XGBRegressor(n_estimators=537, max_depth=15, learning_rate=0.07, min_child_weight=8,
    #                        subsample=0.808, gamma=2.303, alpha=0.698, reg_lambda=5.009,
    #                         objective=xgbloss, tree_method="gpu_hist")
    n_rounds = 1
    n_points_regression = 10000
    use_log_transform = False
    run_feature_importance = False
    run_umap_tsne = False
    run_shap = False
    features = featuresBig

# Import the training dataset
glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)

# Replace zeros
glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, 'THICKNESS'] = glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, ['ith_m', 'ith_f']].mean(axis=1, skipna=True)
# Remove zeros
#glathida_rgis = glathida_rgis[glathida_rgis['THICKNESS'] != 0.0]

glathida_rgis_zeros = glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0.0]

glathida_nan = glathida_rgis[glathida_rgis.isna().any(axis=1)]

fig, ax = plt.subplots()
s1 = ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], s=2, c=glathida_rgis['THICKNESS'])
s2 = ax.scatter(x=glathida_nan['POINT_LON'], y=glathida_nan['POINT_LAT'], s=2, c='r')#c=np.sqrt(glathida_nan['vx']**2+glathida_nan['vy']**2))
cbar = plt.colorbar(s1)
plt.show()


# Filter out some portion of it
#glathida_rgis = glathida_rgis.loc[glathida_rgis['THICKNESS']>=CFG.min_thick_value_train]
#glathida_rgis = glathida_rgis.loc[(glathida_rgis['RGI'] == 3) | (glathida_rgis['RGI'] == 7)]
#glathida_rgis = glathida_rgis[~glathida_rgis['RGI'].isin([19])]
#glathida_rgis = glathida_rgis[glathida_rgis['RGI'].isin([19])]

# Add some features
glathida_rgis['lats'] = glathida_rgis['POINT_LAT']
glathida_rgis['slope50'] = np.sqrt(glathida_rgis['slope_lon_gf50']**2 + glathida_rgis['slope_lat_gf50']**2)
glathida_rgis['slope75'] = np.sqrt(glathida_rgis['slope_lon_gf75']**2 + glathida_rgis['slope_lat_gf75']**2)
glathida_rgis['slope100'] = np.sqrt(glathida_rgis['slope_lon_gf100']**2 + glathida_rgis['slope_lat_gf100']**2)
glathida_rgis['slope125'] = np.sqrt(glathida_rgis['slope_lon_gf125']**2 + glathida_rgis['slope_lat_gf125']**2)
glathida_rgis['slope150'] = np.sqrt(glathida_rgis['slope_lon_gf150']**2 + glathida_rgis['slope_lat_gf150']**2)
glathida_rgis['slope300'] = np.sqrt(glathida_rgis['slope_lon_gf300']**2 + glathida_rgis['slope_lat_gf300']**2)
glathida_rgis['slope450'] = np.sqrt(glathida_rgis['slope_lon_gf450']**2 + glathida_rgis['slope_lat_gf450']**2)
glathida_rgis['slopegfa'] = np.sqrt(glathida_rgis['slope_lon_gfa']**2 + glathida_rgis['slope_lat_gfa']**2)
glathida_rgis['elevation_from_zmin'] = glathida_rgis['elevation'] - glathida_rgis['Zmin']
#glathida_rgis['hbahrm'] = 0.03*(glathida_rgis['Area']**0.375)*1000 # Bahr's approximation: h in meters
#glathida_rgis['hbahrm2'] = glathida_rgis['dist_from_border_km_geom']*np.sqrt(glathida_rgis['Area'])
A = 24 * np.power(10, -25.0) #s−1 Pa−3
rho, g = 917., 9.81
#glathida_rgis['sia'] = ((glathida_rgis['v100']*(3+1))/(2*A*(rho*g*glathida_rgis['slope100'])**3))**(1./4)
glathida_rgis['sia'] = glathida_rgis['v100']/(glathida_rgis['slope100']**3)

# Remove nans (this is an overkill - i want ideally to remove nans only in the training features)
glathida_rgis = glathida_rgis.dropna()
#glathida_rgis = glathida_rgis.dropna(subset=CFG.features + ['THICKNESS'])
print(len(glathida_rgis))

print(f"Overall dataset: {len(glathida_rgis)} rows, {glathida_rgis['RGI'].value_counts()} regions and {glathida_rgis['RGIId'].nunique()} glaciers.")

# umap and tsne
if CFG.run_umap_tsne:
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


def create_test(df, rgi=None, frac=0.1, full_shuffle=True, seed=None):
    """
    - rgi se voglio creare il test in una particolare regione
    - frac: quanto lo voglio grande in percentuale alla grandezza del rgi
    """
    if seed is not None:
        random.seed(seed)

    if rgi is not None and full_shuffle is True:
        df_rgi = df[df['RGI'] == rgi]
        test = df_rgi.sample(frac=frac, random_state=seed)
        train = df.drop(test.index)
        return train, test

    if full_shuffle is True:
        test = df.sample(frac=frac, random_state=seed)
        train = df.drop(test.index)
        return train, test

    # create test based on rgi
    if rgi is not None:
        df_rgi = df[df['RGI']==rgi]
    else:
        df_rgi = df

    minimum_test_size = round(frac * len(df_rgi))

    unique_glaciers = df_rgi['RGIId'].unique()
    random.shuffle(unique_glaciers)
    selected_glaciers = []
    n_total_points = 0
    #print(unique_glaciers)

    for glacier_name in unique_glaciers:
        if n_total_points < minimum_test_size:
            selected_glaciers.append(glacier_name)
            n_points = df_rgi[df_rgi['RGIId'] == glacier_name].shape[0]
            n_total_points += n_points
            #print(glacier_name, n_points, n_total_points)
        else:
            #print('Finished with', n_total_points, 'points, and', len(selected_glaciers), 'glaciers.')
            break

    test = df_rgi[df_rgi['RGIId'].isin(selected_glaciers)]
    train = df.drop(test.index)
    #print(test['RGI'].value_counts())
    #print(test['RGIId'].value_counts())
    #print('Total test size: ', len(test))
    #print(train.describe().T)
    #input('wait')
    return train, test

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
        # To select which parameters to optimize, please look at the XGBoost documentation:
        # https://xgboost.readthedocs.io/en/latest/parameter.html
        "objective": CFG.xgb_loss,
        "n_estimators": trial.suggest_int("n_estimators", 1, 2000),
        "verbosity": 0,
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 7.0, 17.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    train, test = create_test(glathida_rgis, rgi=None, full_shuffle=True, frac=.2, seed=42) #important to decide
    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]

    model = xgb.XGBRegressor(**params, tree_method="gpu_hist")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
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

stds_ML, meds_ML, slopes_ML, rmses_ML = [], [], [], []
stds_Mil, meds_Mil, slopes_Mil, rmses_Mil = [], [], [], []
stds_Far, meds_Far, slopes_Far, rmses_Far = [], [], [], []

best_model = None
best_slope = -999
best_rmse = 9999

for i in range(CFG.n_rounds):

    # Train, val, and test
    train, test = create_test(glathida_rgis, rgi=1, full_shuffle=True, frac=.1, seed=None)

    create_val = False
    if create_val:
        val = glathida_rgis.drop(test.index).sample(n=500)
        train = glathida_rgis.drop(test.index).drop(val.index)

    print(f"Iter {i} Train/Test: {len(train)}/{len(test)}, Train no. glaciers: {train['RGIId'].nunique()}, Test no. glaciers: {test['RGIId'].nunique()}")

    plot_train_test = False
    if plot_train_test:
        fig, ax = plt.subplots()
        s1 = ax.scatter(x=train['POINT_LON'], y=train['POINT_LAT'], s=10, c='b')
        s2 = ax.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=5, c='r')
        plt.show()

    plot_some_train_features = False
    if plot_some_train_features:
        print(train['sia'].describe())
        fig, ax = plt.subplots()
        #ax.hist(train['slope50'], bins=np.arange(train['slope50'].min(), train['slope50'].max(), 0.01), color='k', ec='k', alpha=.4)
        ax.hist(train['sia'], bins=200, color='k', ec='k', alpha=.4)
        plt.show()

    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]
    #print(f'Dataset sizes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')
    y_test_m = test[CFG.millan]
    y_test_f = test[CFG.farinotti]

    ### initializa the model
    model = xgb.XGBRegressor(**CFG.xgb_params, objective=CFG.xgb_loss, tree_method="gpu_hist")
    #model = CFG.model

    if CFG.use_log_transform:
        # Log transform the target variable
        y_train_log = np.log1p(y_train)
        y_test_log = np.log1p(y_test)
        model.fit(X_train, y_train_log)
        y_preds_log = model.predict(X_test)
        y_preds = np.expm1(y_preds_log)

    else:
        #model.fit(X_train, y_train)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
        y_preds = model.predict(X_test)

        # Shap Analysis
        if CFG.run_shap:
            #explainer = shap.Explainer(model)
            explainer = shap.explainers.GPUTree(model, X_train)
            shap_values = explainer(X_train.sample(2000))
            shap.plots.bar(shap_values, max_display=len(CFG.features))
            #shap.plots.beeswarm(shap_values, max_display=len(CFG.features))
            plt.show()


    # Run feature importance
    if CFG.run_feature_importance:
        lgb.plot_importance(model, importance_type="gain", figsize=(7,6), title="LightGBM Feature Importance")
        plt.show()

    # benchmarks
    model_metrics = compute_scores(y_test, y_preds, verbose=False)

    #*** Note: here it is very important since this is the policy to decide which model will be selected for deploy
    rmse = model_metrics['RMSE']
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

    # stats
    mu_ML = np.mean(y_test-y_preds)
    med_ML = np.median(y_test-y_preds)
    std_ML = np.std(y_test-y_preds)
    rmse_ML = np.sqrt(mean_squared_error(y_test, y_preds))

    mu_millan = np.mean(y_test-y_test_m)
    med_millan = np.median(y_test-y_test_m)
    std_millan = np.std(y_test-y_test_m)
    if np.isnan(y_test_m).all() is True:
        rmse_millan = np.nan
    else:
        rmse_millan = np.sqrt(mean_squared_error(y_test, y_test_m))

    mu_farinotti = np.mean(y_test-y_test_f)
    med_farinotti = np.median(y_test-y_test_f)
    std_farinotti = np.std(y_test-y_test_f)
    if np.isnan(y_test_f).all() is True:
        rmse_farinotti = np.nan
    else:
        rmse_farinotti = np.sqrt(mean_squared_error(y_test, y_test_f))

    print(f'{i} Benchmarks ML, Millan and Farinotti: {rmse_ML:.2f} {rmse_millan:.2f} {rmse_farinotti:.2f}')

    # fits
    m_ML, q_ML, _, _, _ = stats.linregress(y_test, y_preds)
    m_millan, q_millan, _, _, _ = stats.linregress(y_test, y_test_m)
    m_farinotti, q_farinotti, _, _, _ = stats.linregress(y_test, y_test_f)

    stds_ML.append(std_ML)
    meds_ML.append(med_ML)
    slopes_ML.append(m_ML)
    rmses_ML.append(rmse_ML)
    stds_Mil.append(std_millan)
    meds_Mil.append(med_millan)
    slopes_Mil.append(m_millan)
    rmses_Mil.append(rmse_millan)
    stds_Far.append(std_farinotti)
    meds_Far.append(med_farinotti)
    slopes_Far.append(m_farinotti)
    rmses_Far.append(rmse_farinotti)

print(f"Res. medians {np.mean(meds_ML):.2f}({np.std(meds_ML):.2f}) {np.mean(meds_Mil):.2f}({np.std(meds_Mil):.2f}) {np.mean(meds_Far):.2f}({np.std(meds_Far):.2f})")
print(f"Res. stdevs {np.mean(stds_ML):.2f}({np.std(stds_ML):.2f}) {np.mean(stds_Mil):.2f}({np.std(stds_Mil):.2f}) {np.mean(stds_Far):.2f}({np.std(stds_Far):.2f})")
print(f"Res. slopes {np.mean(slopes_ML):.2f}({np.std(slopes_ML):.2f}) {np.mean(slopes_Mil):.2f}({np.std(slopes_Mil):.2f}) {np.mean(slopes_Far):.2f}({np.std(slopes_Far):.2f})")
print(f"Rmse {np.mean(rmses_ML):.2f}({np.std(rmses_ML):.2f}) {np.mean(rmses_Mil):.2f}({np.std(rmses_Mil):.2f}) {np.mean(rmses_Far):.2f}({np.std(rmses_Far):.2f})")
print(f"Rmse {100*(np.nanmean(rmses_Mil)-np.nanmean(rmses_ML))/np.nanmean(rmses_Mil):.1f}% better than Millan")
print(f"Rmse {100*(np.nanmean(rmses_Far)-np.nanmean(rmses_ML))/np.nanmean(rmses_Far):.1f}% better than Farinotti")

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
text_ml = f"ML\n$\\mu$ = {mu_ML:.1f}\nmed = {med_ML:.1f}\n$\\sigma$ = {std_ML:.1f}"
text_millan = f"Millan\n$\\mu$ = {mu_millan:.1f}\nmed = {med_millan:.1f}\n$\\sigma$ = {std_millan:.1f}"
text_farinotti = f"Farinotti\n$\\mu$ = {mu_farinotti:.1f}\nmed = {med_farinotti:.1f}\n$\\sigma$ = {std_farinotti:.1f}"
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
        oggm_rgi_shp = glob(f"{args.oggm}rgi/RGIV62/{rgi}*/{rgi}*.shp")[0]
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
# Train the model with full set of available data
# *********************************************
# Import the training dataset
glathida_full = pd.read_csv(args.metadata_file, low_memory=False)

# Replace zeros
glathida_full.loc[glathida_full['THICKNESS'] == 0, 'THICKNESS'] = glathida_full.loc[glathida_full['THICKNESS'] == 0, ['ith_m', 'ith_f']].mean(axis=1, skipna=True)

#glathida_full = glathida_full[~glathida_full['RGI'].isin([19])]
#glathida_full = glathida_full[glathida_full['RGIId'] != 'RGI60-05.13567']

glathida_full['lats'] = glathida_full['POINT_LAT']
#glathida_full['v50'] = np.sqrt(glathida_full['vx_gf50']**2 + glathida_full['vy_gf50']**2)
#glathida_full['v100'] = np.sqrt(glathida_full['vx_gf100']**2 + glathida_full['vy_gf100']**2)
#glathida_full['v150'] = np.sqrt(glathida_full['vx_gf150']**2 + glathida_full['vy_gf150']**2)
#glathida_full['v300'] = np.sqrt(glathida_full['vx_gf300']**2 + glathida_full['vy_gf300']**2)
#glathida_full['v450'] = np.sqrt(glathida_full['vx_gf450']**2 + glathida_full['vy_gf450']**2)
#glathida_full['vgfa'] = np.sqrt(glathida_full['vx_gfa']**2 + glathida_full['vy_gfa']**2)
glathida_full['slope50'] = np.sqrt(glathida_full['slope_lon_gf50']**2 + glathida_full['slope_lat_gf50']**2)
glathida_full['slope75'] = np.sqrt(glathida_full['slope_lon_gf75']**2 + glathida_full['slope_lat_gf75']**2)
glathida_full['slope100'] = np.sqrt(glathida_full['slope_lon_gf100']**2 + glathida_full['slope_lat_gf100']**2)
glathida_full['slope125'] = np.sqrt(glathida_full['slope_lon_gf125']**2 + glathida_full['slope_lat_gf125']**2)
glathida_full['slope150'] = np.sqrt(glathida_full['slope_lon_gf150']**2 + glathida_full['slope_lat_gf150']**2)
glathida_full['slope300'] = np.sqrt(glathida_full['slope_lon_gf300']**2 + glathida_full['slope_lat_gf300']**2)
glathida_full['slope450'] = np.sqrt(glathida_full['slope_lon_gf450']**2 + glathida_full['slope_lat_gf450']**2)
glathida_full['slopegfa'] = np.sqrt(glathida_full['slope_lon_gfa']**2 + glathida_full['slope_lat_gfa']**2)
glathida_full['elevation_from_zmin'] = glathida_full['elevation'] - glathida_full['Zmin']
#glathida_full['sia'] = ((glathida_full['v100']*(3+1))/(2*A*(rho*g*glathida_full['slope100'])**3))**(1./4)
glathida_full['sia'] = glathida_full['v100']/(glathida_full['slope100']**3)


# Remove nans (that remained from trying to replace zeros)
glathida_full = glathida_full.dropna(subset=CFG.features + ['THICKNESS'])

print(f"Full dataset: {len(glathida_full)} rows, {glathida_full['RGI'].value_counts()} regions and {glathida_full['RGIId'].nunique()} glaciers.")
#print('nans in full dataset: ', glathida_full.isna().sum())


stds_ML, meds_ML, slopes_ML, rmses_ML, maes_ML = [], [], [], [], []
best_model = None
best_slope = -999
best_rmse = 9999

for i in range(CFG.n_rounds):

    # Train, val, and test
    train, test = create_test(glathida_full, rgi=None, full_shuffle=True, frac=.1, seed=None)

    create_val = False
    if create_val:
        val = glathida_full.drop(test.index).sample(n=500)
        train = glathida_full.drop(test.index).drop(val.index)

    print(f"Iter {i} Train/Test: {len(train)}/{len(test)}")

    y_train, y_test = train[CFG.target], test[CFG.target]
    X_train, X_test = train[CFG.features], test[CFG.features]
    print(f'Dataset sizes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')

    ### initializa the model
    model = xgb.XGBRegressor(**CFG.xgb_params, objective=CFG.xgb_loss, tree_method="gpu_hist")
    #model = CFG.model

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    #model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    # benchmarks
    model_metrics = compute_scores(y_test, y_preds, verbose=True)

    # *** Note: here it is very important since this is the policy to decide which model will be selected for deploy
    # This is critical. If I choose the test set in a way that it is not random but has only glaciers with shallow
    # thickness, the rmse will be much smaller and I will select the model for this particular dataset.
    # I should have a mechanism that selects the best model based not on the choice of the test dataset.
    # E.g. with create_test full_shuffle=False, RGI60-05.13501 glacier with and without its own points is predicted
    # much differently.
    rmse = model_metrics['RMSE']
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

    # stats
    mu_ML = np.mean(y_test - y_preds)
    med_ML = np.median(y_test - y_preds)
    std_ML = np.std(y_test - y_preds)
    rmse_ML = model_metrics['RMSE']
    mae_ML = model_metrics['MAE']

    print(f'{i} Full data Benchmarks ML: {rmse_ML:.2f}')

    # fits
    m_ML, q_ML, _, _, _ = stats.linregress(y_test, y_preds)

    stds_ML.append(std_ML)
    meds_ML.append(med_ML)
    slopes_ML.append(m_ML)
    rmses_ML.append(rmse_ML)
    maes_ML.append(mae_ML)

print(f"Full data Res. medians {np.mean(meds_ML):.2f}({np.std(meds_ML):.2f})")
print(f"Full data Res. stdevs {np.mean(stds_ML):.2f}({np.std(stds_ML):.2f})")
print(f"Full data Res. slopes {np.mean(slopes_ML):.2f}({np.std(slopes_ML):.2f})")
print(f"Full data Rmse {np.mean(rmses_ML):.2f}({np.std(rmses_ML):.2f})")
print(f"Full data Mae {np.mean(maes_ML):.2f}({np.std(maes_ML):.2f})")

print(f"At the end of cv the best full data rmse is {best_rmse}")
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

glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-03.01710', rgi=3, area=20, seed=None)
#'RGI60-13.37753', RGI60-13.43528 RGI60-13.54431
#glacier_name_for_generation = 'RGI60-07.00228' #RGI60–07.00027 'RGI60-11.01450' RGI60-07.00552,
#glacier_name_for_generation = 'RGI60-07.00832' very nice
# RGI60-03.00832 la differenza tra i modelli è molto interessante
# RGI60-03.01708
#'RGI60-03.01632', 'RGI60-07.01482' ML simile agli altri 2 in termini di alte profondita
# 'RGI60-03.00251' Dobbin Bay, 'RGI60-07.00124' Renardbreen, 'RGI60-11.01328' Unteraargletscher, 'RGI60-11.01478'
# 'RGI60-03.02469', 'RGI60-03.01483 ML << Millan/Farinotti. This is super interesting. These are marine-term glaciers,
# I remember Millan mentioning that marine term glaciers have a bias towards bigger thickness (high speed, low slope)
# 'RGI60-03.01466' RGI60-04.05745 << M-F
# 'RGI60-03.00228'
# Barnes Ice Cap e' molto interessante confrontare le predizioni perche' c'e' area senza ghiaccio attorno per confrontare
# i bedrock! E i volumi di ghiaccio sono enormi
# 'RGI60-11.01492' we can see millan's effect of velocity products on ice thickness calculation
# RGI60-07.00027 biggest in Svalbard
# RGI60-03.01710 biggest in Arctic Canada (Wykeham Glacier South)
# 'RGI60-07.01464' Holtedahlfonna
# in 'RGI60-08.01657' and RGI60-08.01641 I see Millan having gaps (in v hence in ith_m).
# no Millan data: RGI60-08.03159, RGI60-08.03084 controlla questo
# RGI60-03.00862 and RGI60-03.04229 have produced points in another utm zone wrt glacier center (issue?)
# RGI60-03.02811 in interesting since on top Millan has no data, so what is the effect or modeling with/without v ?
#RGI60-07.00174 Farinotti has a gap ?
# RGI60-07.01575 has no millan data
# RGI60-13.54431
# 'RGI60-14.06794' Baltoro glacier
# 'RGI60-04.04988' potrebbe essere emblematico di quanto Far sovrastimi ?
# RGI60-04.05758, RGI60-04.05748 look at the small spatial scales which ML can appreciated (look in relation to hillshade)
# 'RGI60-05.11268' has tandem-x with bugs. The bugs are reflected in my solution (since the model strongly uses the slope)
# RGI60-05.10988 I think Millan and Farinotti solutions are wrong.
# RGI60-05.10743 Farinotti looks badly wrong.

# Generate points for one glacier
test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
                                              n=CFG.n_points_regression, seed=None, verbose=True)
test_glacier_rgi = glacier_name_for_generation[6:8]

# Add features just in case
test_glacier['sia'] = ((0.3*test_glacier['v50']*(3+1))/(2*A*(rho*g*test_glacier['slope50'])**3))**(1./4)

X_test_glacier = test_glacier[CFG.features]
y_test_glacier_m = test_glacier[CFG.millan]  # Note that here nans are present if Millan has no data
y_test_glacier_f = test_glacier[CFG.farinotti]

no_millan_data = np.isnan(y_test_glacier_m).all()
no_farinotti_data = np.isnan(y_test_glacier_f).all()

if CFG.use_log_transform:
    y_preds_glacier_log = best_model.predict(X_test_glacier)
    y_preds_glacier = np.expm1(y_preds_glacier_log)  # Inverse log transform
else:
    y_preds_glacier = best_model.predict(X_test_glacier)

# Set negative predictions to zero
y_preds_glacier = np.where(y_preds_glacier < 0, 0, y_preds_glacier)

# Calculate the bedrock elevations
bedrock_elevations_ML = test_glacier['elevation'] - y_preds_glacier
bedrock_elevations_Millan = test_glacier['elevation'] - y_test_glacier_m
bedrock_elevations_Far = test_glacier['elevation'] - y_test_glacier_f

# Begin to extract all necessary things to plot the result
oggm_rgi_shp = glob(f"{args.oggm}rgi/RGIV62/{test_glacier_rgi}*/{test_glacier_rgi}*.shp")[0]
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
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


# Get Farinotti file and calculate the volume
if not no_farinotti_data:
    test_glacier_folder_farinotti = glob(f"{args.farinotti_icethickness_folder}/*{test_glacier_rgi}/")[0]
    ice_farinotti = rioxarray.open_rasterio(test_glacier_folder_farinotti+glacier_name_for_generation+'_thickness.tif')
    res_farinotti = ice_farinotti.rio.resolution()[0]
    vol_farinotti = 1.e-9 * (res_farinotti ** 2) * np.nansum(ice_farinotti.values) # Volume Farinotti km3
else:
    print(f"Farinotti glacier {glacier_name_for_generation} not found in OGGM V62 database.")
    vol_farinotti = np.nan
# Calculate the glacier volume using the 3 models
vol_ML = calc_volume_glacier(y_preds_glacier, glacier_area)
vol_millan = calc_volume_glacier(y_test_glacier_m, glacier_area)
vol_farinotti2 = calc_volume_glacier(y_test_glacier_f, glacier_area)
print(f"Glacier {glacier_name_for_generation} Area: {glacier_area:.2f} km2, "
      f"volML: {vol_ML:.4g} km3 "
      f"volMil: {vol_millan:.4g} km3 "
      f"volFar: {vol_farinotti:.4g} km3 volFar2: {vol_farinotti2:.4g} km3"
      f"Far mismatch {100*abs(vol_farinotti2-vol_farinotti)/vol_farinotti:.2f}%")

print(f"No. points: {len(y_preds_glacier)} no. positive preds {100*np.sum(y_preds_glacier > 0)/len(y_preds_glacier):.1f}")

# Visualize test predictions of specific glacier
y_min = min(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))
y_max = max(np.concatenate((y_preds_glacier, y_test_glacier_m, y_test_glacier_f)))

create_tif_file = False
if create_tif_file:
    lons = test_glacier['lons'].to_numpy()
    lats = test_glacier['lats'].to_numpy()

    # Create the grid
    grid_res = 0.001  # Adjust as needed
    lat_grid = np.arange(lats.min(), lats.max(), grid_res)
    lon_grid = np.arange(lons.min(), lons.max(), grid_res)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # Interpolate the z values onto the grid
    z_grid = griddata((lons, lats), y_preds_glacier, (lon_grid, lat_grid), method='linear')

    data_array = xarray.DataArray(z_grid, coords=[lat_grid[:, 0], lon_grid[0, :]], dims=['lat', 'lon'])
    data_array = data_array.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    data_array = data_array.rio.write_crs("EPSG:4326")
    data_array.rio.to_raster("ex.tif")

    plt.imshow(z_grid, cmap='jet')
    plt.show()

plot_fancy_ML_prediction = True
if plot_fancy_ML_prediction:
    fig, ax = plt.subplots(figsize=(8,6))

    x0, y0, x1, y1 = exterior_ring.bounds
    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
    hillshade = hillshade.rio.clip_box(minx=x0-dx/2, miny=y0-dy/2, maxx=x1+dx/2, maxy=y1+dy/2)

    im = hillshade.plot(ax=ax, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
    #im = ax.imshow(hillshade, cmap='grey', alpha=0.9, zorder=0)
    #im = ax.imshow(hillshade, cmap='grey', vmin=np.nanmin(focus), vmax=np.nanmax(focus), alpha=0.15, zorder=0)
    vmin = min(y_preds_glacier) # 0 # min(y_preds_glacier)#test_glacier['smb'].min()
    vmax = max(y_preds_glacier) #1600 #max(y_preds_glacier)#test_glacier['smb'].max()
    # y_preds_glacier
    s1 = ax.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=y_preds_glacier,
                     cmap='jet', label='ML', zorder=1, vmin=vmin,vmax=vmax)
    s_glathida = ax.scatter(x=glathida_full['POINT_LON'], y=glathida_full['POINT_LAT'], c=glathida_full['THICKNESS'],
                            cmap='jet', ec='k', s=40, vmin=vmin,vmax=vmax)

    cbar = plt.colorbar(s1, ax=ax)
    cbar.mappable.set_clim(vmin=vmin,vmax=vmax)
    cbar.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=14)
    #cbar.set_label(r'mass balance (mm w.e. yr$^{-1}$)', labelpad=15, rotation=90, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.plot(*exterior_ring.xy, c='k')
    for nunatak in glacier_nunataks_list:
        ax.plot(*nunatak.xy, c='k', lw=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Lon ($^{\\circ}$)', fontsize=14)
    ax.set_ylabel('Lat ($^{\\circ}$)', fontsize=14)
    ax.set_title('')
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    #plt.savefig('/home/maffe/Downloads/RGI60-1313574_CCAI.png', dpi=200)
    plt.show()

# Plot comparison between ML, Millan and Farinotti
fig, axes = plt.subplots(1,3, figsize=(8,4))
ax1, ax2, ax3 = axes.flatten()
s1 = ax1.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_preds_glacier, cmap='jet', label='ML')
#cntr1 = ax1.tricontourf(test_glacier['lons'], test_glacier['lats'], y_preds_glacier, cmap="jet")
if not no_millan_data:
    s2 = ax2.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_m, cmap='jet',
                     label='Millan')
    #cntr2 = ax2.tricontourf(test_glacier['lons'], test_glacier['lats'], y_test_glacier_m>=0, levels=5, cmap="plasma")
if not no_farinotti_data:
    s3 = ax3.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=2, c=y_test_glacier_f, cmap='jet', label='Farinotti')
#cntr3 = ax3.tricontourf(test_glacier['lons'], test_glacier['lats'], y_test_glacier_f>=0, levels=5, cmap="plasma")

ax1.set_title(f"ML {vol_ML:.4g} km3")
ax2.set_title(f"Millan {vol_millan:.4g} km3")
ax3.set_title(f"Farinotti {vol_farinotti:.4g} km3")

cbar1 = plt.colorbar(s1, ax=ax1)
cbar1.mappable.set_clim(vmin=y_min,vmax=y_max)
cbar1.set_label('Thickness (m)', labelpad=15, rotation=270)
if not no_millan_data:
    cbar2 = plt.colorbar(s2, ax=ax2)
    cbar2.mappable.set_clim(vmin=y_min, vmax=y_max)
    cbar2.set_label('Thickness (m)', labelpad=15, rotation=270)
if not no_farinotti_data:
    cbar3 = plt.colorbar(s3, ax=ax3)
    cbar3.mappable.set_clim(vmin=y_min,vmax=y_max)
    cbar3.set_label('Thickness (m)', labelpad=15, rotation=270)

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

####################################
# Regional simulation
def run_rgi_simulation(rgi):
    print("Begin regional simulation ")
    oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')
    # Get glaciers and order them in decreasing order by Area. First glaciers will be bigger and slower to process.
    oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp).sort_values(by='Area', ascending=False)
    # Main loop over glaciers
    for i, gl_id in tqdm(enumerate(oggm_rgi_glaciers['RGIId']), total=len(oggm_rgi_glaciers), desc=f"rgi {rgi} Glaciers", leave=True):
        # Fetch glacier data
        glacier_data = populate_glacier_with_metadata(glacier_name=gl_id, n=CFG.n_points_regression, seed=None, verbose=True)
    print(f"Finished regional simulation for rgi {rgi}.")

run_rgi_simulation_YN = False
if run_rgi_simulation_YN:
    run_rgi_simulation(5)