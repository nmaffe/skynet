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
    L2_penalty=0.000

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
            y=torch.tensor(glathida_rgis[CFG.target].to_numpy().reshape(-1, 1), dtype=torch.float32))

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

    def forward(self, data):
        h, edge_index = data.x, data.edge_index

        h = torch.relu(self.conv1(h, edge_index))
        h = nn.Dropout(0.5)(h)
        h = torch.relu(self.conv2(h, edge_index))
        h = nn.Dropout(0.2)(h)
        h = torch.relu(self.conv3(h, edge_index))
        h = nn.Dropout(0.2)(h)
        h = self.conv4(h, edge_index)
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

test_mask_bool = (glathida_rgis['POINT_LAT']<76) #1879

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


for epoch in range(CFG.epochs):
    model.train()
    optimizer.zero_grad()
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

    print(f'Epoch {epoch} | Train loss {loss:.2f} | Val loss {loss_val:.2f} | Train rmse {rmse:.3f} | Val rmse {rmse_val:.3f}')
