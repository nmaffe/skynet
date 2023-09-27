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
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv, GATConv


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
    epochs =100
    loss = nn.MSELoss()
    L2_penalty=0.000

PATH_METADATA = '/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/'
file = 'TTT_final_grid_20.csv'

glathida_rgis = pd.read_csv(PATH_METADATA+file, low_memory=False)
glathida_rgis = glathida_rgis.loc[glathida_rgis['RGI']==11]
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
            y=torch.tensor(glathida_rgis[CFG.target].to_numpy()), dtype=torch.float32)
print(data)
print(f'Check if data is OK: {data.validate(raise_on_error=True)}')
print(f'Keys: {data.keys}')
print(f'Num nodes: {data.num_nodes}')
print(f'Num edges: {data.num_edges}')
print(f'Num features: {data.num_node_features}')
print(f'Isolated nodes: {data.has_isolated_nodes()}')
print(f'Self loops: {data.has_self_loops()}')
print(f'Directed: {data.is_directed()}')
#data = data.shuffle() # Shuffle (?)
#loader = DataLoader([data], batch_size=32, shuffle=True)

#for batch in loader:
#    print(batch)
    #print(batch.num_graphs)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 100)
        self.conv2 = GCNConv(100, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = Adam(model.parameters(), lr=CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)

model.train()
for epoch in range(5):
    optimizer.zero_grad()
    out = model(data)
    print(out.shape)
    #loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #loss.backward()
    #optimizer.step()