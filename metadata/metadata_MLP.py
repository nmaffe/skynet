import argparse, time
import random
from tqdm import tqdm
import copy, math
import numpy as np
import matplotlib.pyplot as plt
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
from utils_metadata import calc_volume_glacier, get_random_glacier_rgiid, create_train_test

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/media/maffe/nvme/glathida/glathida-3.1.0/"
                        +"glathida-3.1.0/data/metadata29_hmineq0.0_tmin20050000_mean_grid_100.csv", help="Training dataset.")
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
                     'curv_50', 'curv_300', 'curv_gfa', 'aspect_50', 'aspect_300', 'aspect_gfa', 'lats', 'dmdtda_hugo',
                     'smb',
                     ]

    featuresBig = featuresSmall + ['v50', 'v100', 'v150', 'v300', 'v450', 'vgfa', ]

    target = 'THICKNESS'
    millan = 'ith_m'
    farinotti = 'ith_f'

    features = featuresBig
    n_points_regression = 50000
    batch_size = 512
    num_workers = 16
    lr = 0.002
    epochs = 300
    loss = nn.MSELoss()
    L2_penalty=0.000

# ====================================================
# Model
# ====================================================
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(p=0.5)

        #self.fc1 = nn.Linear(len(CFG.features), 100)
        #self.fc1_1 = nn.Linear(100, 50)
        #self.fc1_2 = nn.Linear(50, 20)
        #self.fc1_3 = nn.Linear(20, 1)
        self.fc1 = nn.Linear(len(CFG.features), 100)
        self.fc1_2 = nn.Linear(len(CFG.features), 100)

        self.fc2 = nn.Linear(100, 70)
        self.fc3 = nn.Linear(70, 40)
        self.fc4 = nn.Linear(40, 20)
        self.fc5 = nn.Linear(20, 1)

        self.fc2m = nn.Linear(2, 10)
        self.fc2m_1 = nn.Linear(10, 1)

        self.fc2f = nn.Linear(2, 10)
        self.fc2f_1 = nn.Linear(10, 1)

        #self.fcm = nn.Linear(2, 1)
        #self.fcf = nn.Linear(2, 1)

        #self.fc2 = nn.Linear(len(CFG.features), 100)
        #self.fc2_1 = nn.Linear(100, 50)
        #self.fc2_2 = nn.Linear(50, 20)
        #self.fc2_3 = nn.Linear(20, 1)
        self.fc_A = nn.Linear(2, 10)
        self.fcA_1 = nn.Linear(10, 1)
        #self.fc_final = nn.Linear(2, 1)
        #self.fc3 = nn.Linear(len(CFG.features), 100)
        #self.fc3_1 = nn.Linear(100, 1)

    #def forward(self, x, m, f):
    def forward(self, x):

        x1 = torch.relu(self.fc1(x))
        x1 = nn.Dropout(0.1)(x1)
        x1 = torch.relu(self.fc2(x1))
        x1 = nn.Dropout(0.1)(x1)
        x1 = torch.relu(self.fc3(x1))
        #x1 = nn.Dropout(0.0)(x1)
        x1 = torch.relu(self.fc4(x1))
        #x1 = nn.Dropout(0.1)(x1)
        x1 = self.fc5(x1)

        #hm = torch.concat((x1, m), 1)
        #hm = torch.relu(self.fc2m(hm))
        #hm = nn.Dropout(0.2)(hm)
        #hm = torch.relu(self.fc2m_1(hm))

        #x2 = torch.relu(self.fc1_2(x))
        #x2 = nn.Dropout(0.5)(x2)
        #x2 = self.fc2_2(x2)

        #hf = torch.concat((x2, f), 1)
        #hf = torch.relu(self.fc2f(hf))
        #hf = nn.Dropout(0.2)(hf)
        #hf = torch.relu(self.fc2f_1(hf))

        #x = torch.concat((hm, hf), 1)
        #x =  torch.relu(self.fc_A(x))
        #x = self.fcA_1(x)

        #x1 = torch.relu(self.fc1_1(x1))
        #x1 = nn.Dropout(0.2)(x1)
        #x1 = torch.relu(self.fc1_2(x1))
        #x1 = nn.Dropout(0.2)(x1)
        #x1 = torch.relu(self.fc1_3(x1))

        #x2 = torch.relu(self.fc2(x))
        #x2 = self.dropout(x2)
        #x2 = torch.relu(self.fc2_1(x2))
        #x2 = nn.Dropout(0.2)(x2)
        #x2 = torch.relu(self.fc2_2(x2))
        #x2 = nn.Dropout(0.2)(x2)
        #x2 = torch.relu(self.fc2_3(x2))
        #x1 = torch.sigmoid(self.fc1_3(x1))
        #print('x1:', x1.shape)

        #x1 = torch.add(x1, m)
        #x1 = torch.add(x1, f)

        #x2 = torch.relu(self.fc2(x))    # (N,100)
        #x2 = self.dropout(x2)
        #x2 = torch.relu(self.fc2_1(x2))
        #x2 = self.dropout(x2)
        #x2 = torch.relu(self.fc2_2(x2))
        #x2 = self.dropout(x2)
        #x2 = self.fc2_3(x2)
        #x2 = torch.sigmoid(self.fc2_3(x2))             # (N,1)
        #print('x2:', x2.shape)

        #xm = torch.mul(m, x1)           # (N,1)
        #xf = torch.mul(f, x2)           # (N,1)
        #print('xm:', xm.shape)
        #print('xf:', xf.shape)

        #x_physics = torch.add(xm, xf)

        #x_physics = torch.concat((m, f, x1), 1)#((xm, xf), 1)
        #x_physics = torch.relu(self.fc_A(x_physics))
        #x_physics = self.fcA_1(x_physics)

        #x1m = torch.concat((m, x1), 1)
        #x2f = torch.concat((f, x2), 1)

        #x1m = self.fcm(x1m)
        #x2f = self.fcf(x2f)

        #x = torch.concat((x1, m, f), 1)
        #x = torch.relu(self.fc_A(x))
        #x = self.fcA_1(x)
        #x3 = torch.relu(self.fc3(x))
        #x3 = self.fc3_1(x3)

        #x_out = torch.add(x_physics, x3)

        #print('x:', x.shape)
        #input('wait')

        return x1#, hm, hf#x_physics

# ====================================================
# Dataset
# ====================================================
class MaffeDataset(Dataset):

    def __init__(self, df, transform=None):

        scaler = StandardScaler()

        self.df = df
        #df[CFG.features] = scaler.fit_transform(df[CFG.features])
        #df[CFG.target] = scaler.fit_transform(df[CFG.target].to_numpy().reshape(-1,1))
        self.X_features = df[CFG.features].to_numpy()
        self.target = df[CFG.target].to_numpy()
        self.millan = df[CFG.millan].to_numpy()
        self.farinotti = df[CFG.farinotti].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        xfeatures = torch.tensor(self.X_features[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        millan = torch.tensor(self.millan[idx], dtype=torch.float32)
        farinotti = torch.tensor(self.farinotti[idx], dtype=torch.float32)

        return xfeatures, target, millan, farinotti


# Import the training dataset
glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)
glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, 'THICKNESS'] = glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, ['ith_m', 'ith_f']].mean(axis=1, skipna=True)

# Add some features
glathida_rgis['lats'] = glathida_rgis['POINT_LAT']
glathida_rgis['elevation_from_zmin'] = glathida_rgis['elevation'] - glathida_rgis['Zmin']

glathida_full = glathida_rgis.dropna(subset=CFG.features + ['THICKNESS'])
#glathida_full = glathida_full[~glathida_full['RGI'].isin([19])]

print(len(glathida_full))

print(f"Overall dataset: {len(glathida_full)} rows, {glathida_full['RGI'].value_counts()} regions and {glathida_full['RGIId'].nunique()} glaciers.")


# Train, val, and test
train, val = create_train_test(glathida_full, rgi=None, full_shuffle=True, frac=.2, seed=None)
print(f'Train: {len(train)}, Val: {len(val)}')

# ====================================================
# model & optimizer
# ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'working with {device}')

model = NeuralNetwork()
model.to(device)

optimizer = Adam(model.parameters(), lr=CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=CFG.L2_penalty)

# ====================================================
# Train loop
# ====================================================
def train_loop(train, val, model, optimizer):

    best_rmse = np.inf
    best_weights = None

    # Datasets
    train_dataset = MaffeDataset(train, transform=None)
    val_dataset = MaffeDataset(val, transform=None)
    #print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                                  num_workers=CFG.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, drop_last=True)

    criterion = CFG.loss

    # ====================================================
    # loop
    # ====================================================
    print('Start training loop.')
    since = time.time()

    train_loss_history, val_loss_history = [], []
    train_metrics_history, val_metrics_history = [], []

    for epoch in range(CFG.epochs):

        train_loss_epoch, val_loss_epoch = [], []
        train_metrics_epoch, val_metrics_epoch = [], []

        model.train()
        for step, (X_train, y_train, y_train_m, y_train_f) in enumerate(train_loader):

            X_train = X_train.to(device)    #(N,15)
            y_train = y_train.reshape(-1, 1).to(device)    # (N,1)
            y_train_m = y_train_m.reshape(-1, 1).to(device)
            y_train_f = y_train_f.reshape(-1, 1).to(device)
            #print(type(X_train), X_train.shape, type(y_train), y_train.shape, X_train.dtype, type(y_train_m), y_train_m.shape)
            #input('wait')

            y_preds = model(X_train) #model(X_train, y_train_m, y_train_f) # (N,1) y_preds, out_m, out_f
            loss3 = criterion(y_preds, y_train)
            #loss1 = criterion(y_preds, out_m)
            #loss2 = criterion(y_preds, out_f)
            loss = loss3 #loss1 + loss2 + 2 * loss3
            #print(step, X_train.shape, y_train.shape, y_preds.shape)
            #input('wait')

            r2 = r2_score(y_preds.detach().cpu().numpy(), y_train.detach().cpu().numpy())
            rmse = mean_squared_error(y_preds.detach().cpu().numpy(), y_train.detach().cpu().numpy(), squared=False)

            train_loss_epoch.append(float(loss))
            train_metrics_epoch.append(rmse)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_history.append(np.mean(train_loss_epoch))
        train_metrics_history.append(np.mean(train_metrics_epoch))

        model.eval()
        for step, (X_test, y_test, y_test_m, y_test_f) in enumerate(val_loader):
            X_test = X_test.to(device)
            y_test = y_test.reshape(-1, 1).to(device)
            y_test_m = y_test_m.reshape(-1, 1).to(device)
            y_test_f = y_test_f.reshape(-1, 1).to(device)
            with torch.no_grad():
                y_preds = model(X_test) #model(X_test, y_test_m, y_test_f) #, out_m, out_f
                loss3 = criterion(y_preds, y_test)
                #loss1 = criterion(y_preds, out_m)
                #loss2 = criterion(y_preds, out_f)
                loss = loss3#loss1 + loss2 + 2 * loss3

            r2 = r2_score(y_preds.detach().cpu().numpy(), y_test.detach().cpu().numpy())
            rmse = mean_squared_error(y_preds.detach().cpu().numpy(), y_test.detach().cpu().numpy(), squared=False)

            val_loss_epoch.append(float(loss))
            val_metrics_epoch.append(rmse)

            #if (step % 1000 == 0):
            #    print('Test', epoch, step, '\t', float(loss), '\t', r2)

            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = copy.deepcopy(model.state_dict())

        val_loss_history.append(np.mean(val_loss_epoch))
        val_metrics_history.append(np.mean(val_metrics_epoch))

        print(f'Epoch {epoch} | Train loss {train_loss_history[-1]:.2f} | Val loss {val_loss_history[-1]:.2f} '
              f'| Train rmse {train_metrics_history[-1]:.3f} | Val rmse {val_metrics_history[-1]:.3f}')

    print('Finished training loop.')
    return train_loss_history, train_metrics_history, val_loss_history, val_metrics_history, best_weights

train_loss_history, train_metrics_history, val_loss_history, val_metrics_history, best_weights = train_loop(train, val, model, optimizer)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(0, len(train_loss_history)), train_loss_history, '--', c='r')
ax1.plot(np.arange(0, len(val_loss_history)), val_loss_history, '--', c='g')
ax2.plot(np.arange(0, len(train_metrics_history)), train_metrics_history, 'r-')
ax2.plot(np.arange(0, len(val_metrics_history)), val_metrics_history, 'g-')
plt.show()

# load the best model
best_model = NeuralNetwork()
best_model.load_state_dict(best_weights)
best_model.eval()

glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-03.01710', rgi=3, area=20, seed=None)
test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation, n=CFG.n_points_regression, seed=None, verbose=True)
test_glacier_rgi = glacier_name_for_generation[6:8]

# run best model on test dataset
X_test_glacier = torch.tensor(test_glacier[CFG.features].to_numpy(), dtype=torch.float32)
y_test_m = test_glacier[CFG.millan].to_numpy()
y_test_f = test_glacier[CFG.farinotti].to_numpy()

y_preds_glacier = best_model(X_test_glacier)
y_preds_glacier = y_preds_glacier.detach().cpu().numpy().squeeze()


fig, ax = plt.subplots()
s1 = ax.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=y_preds_glacier,
                     cmap='jet', label='ML', zorder=1)#, vmin=0, vmax=850)
cbar = plt.colorbar(s1)
plt.show()

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

plot_fancy_ML_prediction = True
if plot_fancy_ML_prediction:
    fig, ax = plt.subplots(figsize=(8,6))

    x0, y0, x1, y1 = exterior_ring.bounds
    dx, dy = x1 - x0, y1 - y0
    hillshade = copy.deepcopy(focus)
    hillshade.values = earthpy.spatial.hillshade(focus, azimuth=315, altitude=0)
    hillshade = hillshade.rio.clip_box(minx=x0-dx/4, miny=y0-dy/4, maxx=x1+dx/4, maxy=y1+dy/4)

    im = hillshade.plot(ax=ax, cmap='grey', alpha=0.9, zorder=0, add_colorbar=False)
    #im = ax.imshow(hillshade, cmap='grey', alpha=0.9, zorder=0)
    #im = ax.imshow(hillshade, cmap='grey', vmin=np.nanmin(focus), vmax=np.nanmax(focus), alpha=0.15, zorder=0)
    vmin = min(y_preds_glacier) # 0 # min(y_preds_glacier)#test_glacier['smb'].min()
    vmax = max(y_preds_glacier) #1600 #max(y_preds_glacier)#test_glacier['smb'].max()
    # y_preds_glacier
    s1 = ax.scatter(x=test_glacier['lons'], y=test_glacier['lats'], s=1, c=y_preds_glacier,
                     cmap='jet', label='ML', zorder=1, vmin=vmin,vmax=vmax)
    s_glathida = ax.scatter(x=glathida_full['POINT_LON'], y=glathida_full['POINT_LAT'], c=glathida_full['THICKNESS'],
                            cmap='jet', ec='grey', lw=0.5, s=20, vmin=vmin,vmax=vmax)

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

# benchmarks
# nn_metrics = evaluation(y_test, y_preds)

