import time
import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from utils import haversine
import pandas as pd
import geopandas as gpd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, QuantileTransformer

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit

import networkx as nx
from torch_geometric.utils import to_networkx


class CFG:

    features = ['Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
     'dist_from_border_km', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
     'elevation_from_zmin', 'RGI']
    millan = 'ith_m'
    farinotti = 'ith_f'
    target = 'THICKNESS'
    batch_size = 512
    num_workers = 16
    lr = 0.001
    epochs = 10000
    loss = nn.MSELoss()
    L2_penalty = 0.0

PATH_METADATA = '/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/'
file = 'TTT_final_grid_20.csv'

glathida_rgis = pd.read_csv(PATH_METADATA+file, low_memory=False)
glathida_rgis = glathida_rgis.loc[glathida_rgis['RGI']==3]
print(f'Dataset: {len(glathida_rgis)} rows and', glathida_rgis['RGIId'].nunique(), 'glaciers.')
print(list(glathida_rgis))

# Calculate pair-wise haversine distances between all points in a vectorized fashion
lon_ar = np.array(glathida_rgis['POINT_LON'])
lat_ar = np.array(glathida_rgis['POINT_LAT'])
distances = haversine(lon_ar, lat_ar, lon_ar[:, np.newaxis], lat_ar[:, np.newaxis])
print(f'Matrix of pair-wise distances: {distances.shape}')

run_check_example = False
if run_check_example:
    n1, n2 = 400, 500
    lon1, lat1 = glathida_rgis['POINT_LON'].iloc[n1], glathida_rgis['POINT_LAT'].iloc[n1]
    lon2, lat2 = glathida_rgis['POINT_LON'].iloc[n2], glathida_rgis['POINT_LAT'].iloc[n2]
    d =  haversine(lon1, lat1, lon2, lat2)
    print(d, distances[n1, n2], distances[n2, n1])

# non-vectorized way
run_non_vectorized = False
if run_non_vectorized:
    distances_nv = np.zeros((len(lats), len(lats)))
    for n, (lon, lat) in enumerate(zip(lon_ar, lat_ar)):
        #print(n, '/', len(lats))
        for i, (i_lon, i_lat) in enumerate(zip(lon_ar, lat_ar)):
            d = haversine(lon, lat, i_lon, i_lat)
            distances_nv[n,i] = d


# Adjacency matrix
# if distances < 3km, edge=1, otherwise edge=0
threshold = 3.0
adj_matrix = np.where(distances < threshold, 1, 0)
#print(adj_matrix)
# remove self-connections (set diagonal to zero)
np.fill_diagonal(adj_matrix, 0)
#print(adj_matrix)


# Edges
# find indexes corresponding to 1 in the adjacency matrix
edge_index = np.argwhere(adj_matrix==1) # (node_indexes, 2)
edge_index = edge_index.transpose() # (2, node_indexes, 2)
print(f'Edge index vector: {edge_index.shape}')

# graph
data = Data(x=torch.tensor(glathida_rgis[CFG.features].to_numpy(), dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(glathida_rgis[CFG.target].to_numpy().reshape(-1, 1), dtype=torch.float32),
            m=torch.tensor(glathida_rgis[CFG.millan].to_numpy().reshape(-1, 1), dtype=torch.float32),
            f=torch.tensor(glathida_rgis[CFG.farinotti].to_numpy().reshape(-1, 1), dtype=torch.float32))

def print_graph_info(grafo):
    print(grafo)
    print(f'Check if data is OK: {grafo.validate(raise_on_error=True)}')
    print(f'Keys: {grafo.keys}')
    print(f'Num nodes: {grafo.num_nodes}')
    print(f'Num edges: {grafo.num_edges}')
    print(f'Num features: {grafo.num_node_features}')
    print(f'Isolated nodes: {grafo.has_isolated_nodes()}')
    print(f'Self loops: {grafo.has_self_loops()}')
    print(f'Directed: {grafo.is_directed()}')

print_graph_info(data)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 100)
        self.conv2 = GCNConv(100, 50)
        self.conv3 = GCNConv(50, 20)
        self.conv4 = GCNConv(20, 1)

        self.fc_A = GCNConv(3, 10)
        self.fcA_1 = GCNConv(10, 1)

    def forward(self, data):
        h, edge_index = data.x, data.edge_index
        m, f = data.m, data.f

        h = torch.relu(self.conv1(h, edge_index))
        h = nn.Dropout(0.2)(h)
        h = torch.relu(self.conv2(h, edge_index))
        h = nn.Dropout(0.1)(h)
        h = torch.relu(self.conv3(h, edge_index))
        h = nn.Dropout(0.1)(h)
        h = self.conv4(h, edge_index)

        x = torch.concat((h, m, f), 1)
        x = torch.relu(self.fc_A(x, edge_index))
        x = self.fcA_1(x, edge_index)

        return h


# Plot some stuff
ifplot = False
if ifplot is True:
    G = to_networkx(data, to_undirected=True)
    nx.draw(G)
    plt.show()


# Train / Val / Test
train_mask_bool = pd.Series(True, index=glathida_rgis.index)
val_mask_bool = pd.Series(False, index=glathida_rgis.index)

test_mask_bool = (glathida_rgis['POINT_LAT']<76)

train_mask_bool[test_mask_bool] = False
print(len(train_mask_bool), train_mask_bool.sum())

some_val_indexes = train_mask_bool[test_mask_bool==False].sample(1000).index
train_mask_bool[some_val_indexes] = False
print(len(train_mask_bool), train_mask_bool.sum())

val_mask_bool[some_val_indexes] = True
print(len(val_mask_bool), val_mask_bool.sum())


print(f'Train / Val / Test : {np.sum(train_mask_bool)} {np.sum(val_mask_bool)} {np.sum(test_mask_bool)}')
data['test_mask'] = torch.tensor(test_mask_bool.to_numpy(), dtype=torch.bool)
data['val_mask'] = torch.tensor(val_mask_bool.to_numpy(), dtype=torch.bool)
data['train_mask'] = torch.tensor(train_mask_bool.to_numpy(), dtype=torch.bool)

#transform = RandomNodeSplit(split='train_rest', num_val=0.1)
#data = transform(data)

print('-'*50)
print_graph_info(data)
print('-'*50)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=CFG.L2_penalty)

best_rmse = np.inf
best_weights = None

for epoch in range(CFG.epochs):
    model.train()
    optimizer.zero_grad(data)

    out = model(data) # NB out contiene TUTTI i nodi
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    rmse = mean_squared_error(out[data.train_mask].detach().cpu().numpy(),
                              data.y[data.train_mask].detach().cpu().numpy(), squared=False)

    model.eval()
    with torch.no_grad():
        out = model(data)
        loss_val = criterion(out[data.val_mask], data.y[data.val_mask])
        rmse_val = mean_squared_error(out[data.val_mask].detach().cpu().numpy(),
                                  data.y[data.val_mask].detach().cpu().numpy(), squared=False)

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_weights = copy.deepcopy(model.state_dict())

    if (epoch%100 == 0):
        print(f'Epoch {epoch} | Train loss {loss:.2f} | Val loss {loss_val:.2f} | Train rmse {rmse:.3f} | Val rmse {rmse_val:.3f}')

print('Finished training loop.')

# Inference on test
best_model = GCN().to(device)
best_model.load_state_dict(best_weights)
best_model.eval()

y_test = data.y[data.test_mask].detach().cpu().numpy().squeeze()
out = best_model(data)
y_preds = out[data.test_mask].detach().cpu().numpy().squeeze()

y_test_m = glathida_rgis[CFG.millan][test_mask_bool==True].to_numpy()
y_test_f = glathida_rgis[CFG.farinotti][test_mask_bool==True].to_numpy()


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
s1 = ax1.scatter(x=y_test, y=y_test_m, s=5, c='g', alpha=.3)
s1 = ax1.scatter(x=y_test, y=y_test_f, s=5, c='r', alpha=.3)
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
