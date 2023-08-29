import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CFG:
    cols = ['RGI', 'POINT_LAT', 'POINT_LON', 'THICKNESS', 'Area', 'elevation_astergdem',
            'slope_lat', 'slope_lon', 'vx', 'vy', 'dist_from_border_km', 'v', 'slope',
            'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'elevation_from_zmin', 'ith_m', 'ith_f']

    features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
            'dist_from_border_km', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'elevation_from_zmin']
    millan = 'ith_m'
    farinotti = 'ith_f'
    target = 'THICKNESS'
    batch_size = 2048
    num_workers = 16
    lr = 0.001
    epochs = 25

# ====================================================
# Model
# ====================================================
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(CFG.features), 100)
        self.fc2 = nn.Linear(100, 1)
    def forward(self, x, m, f):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x

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


PATH_METADATA = '/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_final.csv'

glathida = pd.read_csv(PATH_METADATA, low_memory=False)

# Add features
glathida['v'] = np.sqrt(glathida['vx']**2 + glathida['vy']**2)
glathida['slope'] = np.sqrt(glathida['slope_lat']**2 + glathida['slope_lon']**2)
glathida['elevation_from_zmin'] = glathida['elevation_astergdem'] - glathida['Zmin']

rgis = [3, 7, 8, 11]

cond = ((glathida['RGI'].isin(rgis)) & (glathida['SURVEY_DATE'] > 20050000)
        & (glathida['DATA_FLAG'].isna()) & (glathida['THICKNESS']>=0)
        & (glathida['ith_m']>=0) & (glathida['ith_f']>=0))

glathida_rgis = glathida[cond]
print('here', list(glathida_rgis))

# Keep only these columns
glathida_rgis = glathida_rgis[CFG.cols]

# Cast all columns to float32: (probably useless)
glathida_rgis = glathida_rgis.astype('float32')

# Shuffle (probably useless)
glathida_rgis = glathida_rgis.sample(frac = 1)

# Remove nans
glathida_rgis = glathida_rgis.dropna(subset=CFG.cols)
print(f'After having removed nans we have {len(glathida_rgis)} rows')
print(glathida_rgis['RGI'].value_counts())

# -----------------------------------------------------------
# Train and Test
val = glathida_rgis.loc[(glathida_rgis['RGI']==3) & (glathida_rgis['POINT_LAT']<76)]
train = glathida_rgis.drop(val.index)
test = copy.deepcopy(val)

# ====================================================
# model & optimizer
# ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'working with {device}')

model = NeuralNetwork()
model.to(device)

optimizer = Adam(model.parameters(), lr=CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)

# ====================================================
# Train loop
# ====================================================
def train_loop(model, optimizer):

    best_mse = np.inf
    best_weights = None

    # Datasets
    train_dataset = MaffeDataset(train, transform=None)
    test_dataset = MaffeDataset(val, transform=None)
    print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                                  num_workers=CFG.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, drop_last=True)

    criterion = nn.MSELoss()

    # ====================================================
    # loop
    # ====================================================

    print('Start training loop.')
    since = time.time()

    train_loss_history, test_loss_history = [], []
    train_metrics_history, test_metrics_history = [], []

    for epoch in range(CFG.epochs):

        train_loss_epoch, test_loss_epoch = [], []
        train_metrics_epoch, test_metrics_epoch = [], []

        model.train()
        for step, (X_train, y_train, y_train_m, y_train_f) in enumerate(train_loader):

            X_train = X_train.to(device)    #(N,15)
            y_train = y_train.reshape(-1, 1).to(device)    # (N,1)
            #print(type(X_train), X_train.shape, type(y_train), y_train.shape, X_train.dtype)

            y_preds = model(X_train, y_train_m, y_train_f) # (N,1)
            loss = criterion(y_preds, y_train)

            #print(step, X_train.shape, y_train.shape, y_preds.shape)

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
        for step, (X_test, y_test, y_test_m, y_test_f) in enumerate(test_loader):
            X_test = X_test.to(device)
            y_test = y_test.reshape(-1, 1).to(device)
            with torch.no_grad():
                y_preds = model(X_test, y_test_m, y_test_f)
                loss = criterion(y_preds, y_test)

            r2 = r2_score(y_preds.detach().cpu().numpy(), y_test.detach().cpu().numpy())
            rmse = mean_squared_error(y_preds.detach().cpu().numpy(), y_test.detach().cpu().numpy(), squared=False)

            test_loss_epoch.append(float(loss))
            test_metrics_epoch.append(rmse)

            #if (step % 1000 == 0):
            #    print('Test', epoch, step, '\t', float(loss), '\t', r2)

            if float(loss) < best_mse:
                best_mse = float(loss)
                best_weights = copy.deepcopy(model.state_dict())

        test_loss_history.append(np.mean(test_loss_epoch))
        test_metrics_history.append(np.mean(test_metrics_epoch))

        print(f'Epoch {epoch} | Train loss {train_loss_history[-1]:.2f} | Val loss {test_loss_history[-1]:.2f} '
              f'| Train rmse {train_metrics_history[-1]:.3f} | Val rmse {test_metrics_history[-1]:.3f}')


    return train_loss_history, train_metrics_history, test_loss_history, test_metrics_history, best_weights

train_loss_history, train_metrics_history, test_loss_history, test_metrics_history, best_weights = train_loop(model, optimizer)

# load the best model
best_model = NeuralNetwork()
best_model.load_state_dict(best_weights)
best_model.eval()

# run best model on test dataset
#scaler1 = StandardScaler()
#test[CFG.features] = scaler1.fit_transform(test[CFG.features])
X_test = torch.tensor(test[CFG.features].to_numpy(), dtype=torch.float32)
y_test = test[CFG.target].to_numpy()
y_test_m = torch.tensor(test[CFG.millan].to_numpy(), dtype=torch.float32)
y_test_f = torch.tensor(test[CFG.farinotti].to_numpy(), dtype=torch.float32)
#mu, devst = np.mean(y_test), np.std(y_test)
y_preds = best_model(X_test, y_test_m, y_test_f).detach().cpu().numpy().squeeze()
#y_preds = y_preds * devst + mu


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

fig, ax = plt.subplots()
s1 = ax.plot(range(len(train_loss_history)), train_loss_history, 'o-', c='b', alpha=.3)
s2 = ax.plot(range(len(test_loss_history)), test_loss_history, 'o-', c='r', alpha=.3)
plt.show()

nn_metrics = evaluation(y_test, y_preds)

fig, (ax1, ax2) = plt.subplots(1,2)
s1 = ax1.scatter(x=y_test, y=y_preds, s=1)
s2 = ax1.plot([0.0, 1200], [0.0, 1200], c='r')
ax1.set_xlabel('Glathida ice thickness (m)')
ax1.set_ylabel('Modelled ice thickness (m)')

ax2.hist(y_test-y_preds, bins=np.arange(-1000, 1000, 100), color='lightblue', ec='blue', alpha=.5)
ax2.text(0.75, 0.9, f'Mean: {np.mean(y_test-y_preds):.2f}', transform = ax2.transAxes)
ax2.text(0.75, 0.85, f'Std: {np.std(y_test-y_preds):.0f}', transform = ax2.transAxes)
ax2.set_xlabel('Glathida - ML model (m)')
plt.show()