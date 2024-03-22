import argparse, time
import copy
from glob import glob
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import geopandas as gpd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, QuantileTransformer

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_final2_grid_20.csv",
                    help="Training dataset.")
parser.add_argument('--save_model', type=bool, default=False, help="True to save the model.")
parser.add_argument('--save_outdir', type=str, default="/home/nico/PycharmProjects/skynet/code/metadata/", help="Saved model dir.")
parser.add_argument('--save_outname', type=str, default="model_mlp_weights.pth", help="Saved model name.")
args = parser.parse_args()

#todo: I now need to add the calculation for v, slope, elevation_from_zmin as not contained in imported metadata file

class CFG:

    #features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
    # 'dist_from_border_km', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
    # 'elevation_from_zmin', 'RGI']
    features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
                'dist_from_border_km_geom', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
                'elevation_from_zmin', 'RGI']

    millan = 'ith_m'
    farinotti = 'ith_f'
    target = 'THICKNESS'
    batch_size = 512
    num_workers = 16
    lr = 0.002
    epochs = 500
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

        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 30)
        self.fc4 = nn.Linear(30, 10)
        self.fc5 = nn.Linear(10, 1)

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
        x1 = nn.Dropout(0.5)(x1)
        x1 = torch.relu(self.fc2(x1))
        x1 = nn.Dropout(0.2)(x1)
        x1 = torch.relu(self.fc3(x1))
        x1 = nn.Dropout(0.1)(x1)
        x1 = torch.relu(self.fc4(x1))
        x1 = nn.Dropout(0.1)(x1)
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



glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)
#glathida_rgis = glathida_rgis.loc[glathida_rgis['THICKNESS'] > 0]
glathida_rgis = glathida_rgis.loc[glathida_rgis['RGI'] == 8]
print(f'Dataset: {len(glathida_rgis)} rows and', glathida_rgis['RGIId'].nunique(), 'glaciers.')


# -----------------------------------------------------------
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
            print(glacier_name, n_points, n_total_points)
        else:
            #print('Finished with', n_total_points, 'points, and', len(selected_glaciers), 'glaciers.')
            break

    test = df[df['RGIId'].isin(selected_glaciers)]
    print(test['RGI'].value_counts())
    print('Total test size: ', len(test))
    return test

# Train, val, and test
test = create_test(glathida_rgis,  minimum_test_size=400, rgi=8, seed=4)
#test = glathida_rgis.loc[(glathida_rgis['RGI']==11) & (glathida_rgis['POINT_LON']<7.2)]
#test = glathida_rgis.loc[(glathida_rgis['RGI']==3) & (glathida_rgis['POINT_LAT']<76)]#.sample(n=1879)
#val = glathida_rgis.loc[(glathida_rgis['RGI']==3)].drop(test.index).sample(n=2000)
val = glathida_rgis.drop(test.index).sample(n=1000)
#train = glathida_rgis.sample(100000)#.drop(val.index)
#train = glathida_rgis.drop(val.index).sample(500000)
train = glathida_rgis.drop(test.index).drop(val.index)

# plot train-val-test
fig, ax = plt.subplots()
im1 = ax.scatter(x=train['POINT_LON'], y=train['POINT_LAT'], c='g', s=1)
im2 = ax.scatter(x=val['POINT_LON'], y=val['POINT_LAT'], c='b', s=1)
im3 = ax.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], c='r', s=1)
plt.show()

print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

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

# run best model on test dataset
#scaler1 = StandardScaler()
#test[CFG.features] = scaler1.fit_transform(test[CFG.features])
X_test = torch.tensor(test[CFG.features].to_numpy(), dtype=torch.float32)
y_test = test[CFG.target].to_numpy()
y_test_m = test[CFG.millan].to_numpy()
y_test_f = test[CFG.farinotti].to_numpy()
#y_preds, out_m, out_f = best_model(X_test,
#                                torch.tensor(y_test_m, dtype=torch.float32).reshape(-1, 1),
#                                torch.tensor(y_test_f, dtype=torch.float32).reshape(-1, 1))
y_preds = best_model(X_test)

y_preds = y_preds.detach().cpu().numpy().squeeze()
#print(y_test.shape, y_preds.shape, y_test_m.shape, y_test_f.shape)

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
nn_metrics = evaluation(y_test, y_preds)

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
m_ML, q_ML, _, _, _ = stats.linregress(y_test,y_preds)
m_millan, q_millan, _, _, _ = stats.linregress(y_test,y_test_m)
m_farinotti, q_farinotti, _, _, _ = stats.linregress(y_test,y_test_f)

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

ax2.hist(y_test-y_preds, bins=np.arange(-1000, 1000, 10), label='NN', color='lightblue', ec='blue', alpha=.4, zorder=2)
ax2.hist(y_test-y_test_m, bins=np.arange(-1000, 1000, 10), label='Millan', color='green', ec='green', alpha=.3, zorder=1)
ax2.hist(y_test-y_test_f, bins=np.arange(-1000, 1000, 10), label='Farinotti', color='red', ec='red', alpha=.3, zorder=1)

# text
text_ml = f'NN\n$\mu$ = {mu_ML:.1f}\nmed = {med_ML:.1f}\n$\sigma$ = {std_ML:.1f}'
text_millan = f'Millan\n$\mu$ = {mu_millan:.1f}\nmed = {med_millan:.1f}\n$\sigma$ = {std_millan:.1f}'
text_farinotti = f'Farinotti\n$\mu$ = {mu_farinotti:.1f}\nmed = {med_farinotti:.1f}\n$\sigma$ = {std_farinotti:.1f}'
# text boxes
props_ML = dict(boxstyle='round', facecolor='lightblue', ec='blue', alpha=0.4)
props_millan = dict(boxstyle='round', facecolor='lime', ec='green', alpha=0.4)
props_farinotti = dict(boxstyle='round', facecolor='salmon', ec='red', alpha=0.4)
ax2.text(0.05, 0.95, text_ml, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_ML)
ax2.text(0.05, 0.7, text_millan, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_millan)
ax2.text(0.05, 0.45, text_farinotti, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=props_farinotti)

ax1.set_xlabel('Glathida ice thickness (m)')
ax1.set_ylabel('Modelled ice thickness (m)')
ax2.set_xlabel('Glathida - Model (m)')
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

s1 = ax1.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test, cmap='Blues', label='Glathida')
s2 = ax2.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_preds, cmap='Blues', label='MLP')
s3 = ax3.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test_m, cmap='Blues', label='Millan')
s4 = ax4.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test_f, cmap='Blues', label='Farinotti')
s5 = ax5.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test-y_preds, cmap='bwr', label='Glathida-MLP')
s6 = ax6.scatter(x=test['POINT_LON'], y=test['POINT_LAT'], s=10, c=y_test-y_test_f, cmap='bwr', label='Glathida-Farinotti')

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
    cbar.mappable.set_clim(vmin=-absmax, vmax=absmax)


for ax in (ax1, ax2, ax3, ax4, ax5, ax6): ax.legend(loc='upper left')

plt.show()
