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
from utils_metadata import calc_volume_glacier, get_random_glacier_rgiid, create_train_test, haversine

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, GATConv
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import to_networkx

import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('--metadata_file', type=str, default="/media/maffe/nvme/glathida/glathida-3.1.0/"
                        +"glathida-3.1.0/data/metadata31_hmineq0.0_tmin20050000_mean_grid_20.csv", help="Training dataset.")
parser.add_argument('--farinotti_icethickness_folder', type=str,default="/media/maffe/nvme/Farinotti/composite_thickness_RGI60-all_regions/",
                    help="Path to Farinotti ice thickness data")
parser.add_argument('--mosaic', type=str,default="/media/maffe/nvme/Tandem-X-EDEM/", help="Path to Tandem-X-EDEM")
parser.add_argument('--oggm', type=str,default="/home/maffe/OGGM/", help="Path to OGGM folder")
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

    n_points_regression = 30000

    batch_size = 512
    num_workers = 16
    lr = 0.002
    epochs = 20000  # 30000#10000
    loss = nn.MSELoss()
    L2_penalty = 0.0
    threshold = .1  # 3.0 #.5 # Penso che al tendere a zero ottengo un mlp
    features = featuresBig


# Import the training dataset
glathida_rgis = pd.read_csv(args.metadata_file, low_memory=False)

# Replace zeros
glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, 'THICKNESS'] = glathida_rgis.loc[glathida_rgis['THICKNESS'] == 0, ['ith_m', 'ith_f']].mean(axis=1, skipna=True)

# Add some features
glathida_rgis['lats'] = glathida_rgis['POINT_LAT']
glathida_rgis['elevation_from_zmin'] = glathida_rgis['elevation'] - glathida_rgis['Zmin']

glathida_rgis = glathida_rgis.dropna(subset=CFG.features + ['THICKNESS'])

glathida_rgis = glathida_rgis.sample(frac=0.5)

print(f"Overall dataset: {len(glathida_rgis)} rows, {glathida_rgis['RGI'].value_counts()} regions and {glathida_rgis['RGIId'].nunique()} glaciers.")


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
adj_matrix = np.where(distances < CFG.threshold, 1, 0)
#print(adj_matrix)
# remove self loops (set diagonal to zero)
np.fill_diagonal(adj_matrix, 0)
#print(adj_matrix)

# Edges
# find indexes corresponding to 1 in the adjacency matrix
edge_index = np.argwhere(adj_matrix==1) # (node_indexes, 2)
edge_index = edge_index.transpose() # (2, node_indexes)
print(f'Edge index vector: {edge_index.shape}')

edge_weight = distances[adj_matrix==1]
print(f'Check min and max should be > 0 and < threshold: {np.min(edge_weight)}, {np.max(edge_weight)}')
edge_weight = 1./(edge_weight ** 1) #np.zeros_like(edge_weight) #

# graph
data = Data(x=torch.tensor(glathida_rgis[CFG.features].to_numpy(), dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
            y=torch.tensor(glathida_rgis[CFG.target].to_numpy().reshape(-1, 1), dtype=torch.float32),
            m=torch.tensor(glathida_rgis[CFG.millan].to_numpy().reshape(-1, 1), dtype=torch.float32),
            f=torch.tensor(glathida_rgis[CFG.farinotti].to_numpy().reshape(-1, 1), dtype=torch.float32)
            )

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

# Plot some stuff
ifplot = False
if ifplot is True:
    #G = to_networkx(data, to_undirected=True)
    #nx.draw(G)
    #plt.show()
    def convert_to_networkx(graph, n_sample=None):

        g = to_networkx(graph, node_attrs=["x"], to_undirected=True)
        y = graph.y.numpy()

        if n_sample is not None:
            sampled_nodes = random.sample(g.nodes, n_sample)
            g = g.subgraph(sampled_nodes)
            y = y[sampled_nodes]

        return g, y


    def plot_graph(g, y):

        plt.figure(figsize=(9, 7))
        nx.draw_spring(g, node_size=30, arrows=False, node_color=y)
        plt.show()


    g, y = convert_to_networkx(data, n_sample=1000)
    plot_graph(g, y)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 100, improved=True)
        #self.conv1 = GraphConv(data.num_node_features, 100, aggr='mean')
        #self.conv1 = GATConv(data.num_node_features, 100)
        #self.conv1_2 = GCNConv(data.num_node_features, 100, improved=False)
        self.conv2 = GCNConv(100, 70, improved=True)
        #self.conv2 = GraphConv(100, 70, aggr='mean')
        #self.conv2 = GATConv(100, 70)
        #self.conv2_2 = GCNConv(100, 1, improved=False)
        self.conv3 = GCNConv(70, 40, improved=True)
        #self.conv3 = GraphConv(70, 40, aggr='mean')
        #self.conv3 = GATConv(70, 40)
        #self.conv3_1 = GCNConv(30, 30, improved=False)
        self.conv4 = GCNConv(40, 20, improved=True)
        #self.conv4 = GraphConv(40, 20, aggr='mean')
        #self.conv4 = GATConv(40, 20)
        self.conv4_1 = GCNConv(20, 1, improved=True)
        #self.conv4_1 = GraphConv(20, 1, aggr='mean')
        #self.conv4_1 = GATConv(20, 1)

        self.fc_A = GCNConv(2, 10, improved=False)
        self.fcA_1 = GCNConv(10, 1, improved=False)

        self.conv2m = GCNConv(2, 10, improved=False)
        self.conv2m_1 = GCNConv(10, 1, improved=False)

        self.conv2f = GCNConv(2, 10, improved=False)
        self.conv2f_1 = GCNConv(10, 1, improved=False)

    def forward(self, data):
        h, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        m, f = data.m, data.f

        #h = torch.concat((h, m, f), 1)
        h = torch.relu(self.conv1(h, edge_index, edge_weight=edge_weight)) #edge_weight=edge_weight
        #h = torch.relu(self.conv1(h, edge_index))
        h = nn.Dropout(0.1)(h)
        h = torch.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        #h = torch.relu(self.conv2(h, edge_index))
        h = nn.Dropout(0.1)(h)
        h = torch.relu(self.conv3(h, edge_index, edge_weight=edge_weight))
        #h = torch.relu(self.conv3(h, edge_index))
        #h = nn.Dropout(0.1)(h)
        #h = torch.relu(self.conv3_1(h, edge_index, edge_weight=edge_weight))
        #h = nn.Dropout(0.1)(h)
        h = torch.relu(self.conv4(h, edge_index, edge_weight=edge_weight))
        #h = torch.relu(self.conv4(h, edge_index))
        #h = nn.Dropout(0.1)(h)
        h = self.conv4_1(h, edge_index, edge_weight=edge_weight)
        #h = self.conv4_1(h, edge_index)

        #h = torch.concat((h, m, f), 1)
        #h = torch.relu(self.fc_A(h, edge_index, edge_weight=edge_weight))
        #x = nn.Dropout(0.1)(x)
        #h = self.fcA_1(h, edge_index, edge_weight=edge_weight)

        '''with Millan and Farinotti'''
        '''
        h1 = torch.relu(self.conv1(h, edge_index))#, edge_attr=edge_weight))
        h1 = nn.Dropout(0.5)(h1)
        h1 = self.conv2(h1, edge_index)#, edge_attr=edge_weight)

        hm = torch.concat((h1, m), 1)
        hm = torch.relu(self.conv2m(hm, edge_index))#, edge_attr=edge_weight))
        hm = nn.Dropout(0.2)(hm)
        hm = torch.relu(self.conv2m_1(hm, edge_index))#, edge_attr=edge_weight)) #(N,1)

        h2 = torch.relu(self.conv1_2(h, edge_index))#, edge_attr=edge_weight))
        h2 = nn.Dropout(0.5)(h2)
        h2 = self.conv2_2(h2, edge_index)#, edge_attr=edge_weight)

        hf = torch.concat((h2, f), 1)
        hf = torch.relu(self.conv2f(hf, edge_index))#, edge_attr=edge_weight))
        hf = nn.Dropout(0.2)(hf)
        hf = torch.relu(self.conv2f_1(hf, edge_index))#, edge_attr=edge_weight))  # (N,1)

        h = torch.concat((hm, hf), 1) #(N,2)
        h = torch.relu(self.fc_A(h, edge_index))#, edge_attr=edge_weight))
        h = self.fcA_1(h, edge_index)#, edge_attr=edge_weight)'''

        return h#, hm, hf


# Train / Val / Test
def create_test_index(df, minimum_test_size=1000, rgi=None, seed=None):
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
    return test.index

train_mask_bool = pd.Series(True, index=glathida_rgis.index)
val_mask_bool = pd.Series(False, index=glathida_rgis.index)
#test_index = create_test_index(glathida_rgis,  minimum_test_size=1800, rgi=7, seed=None)
#test_mask_bool = pd.Series(glathida_rgis.index.isin(test_index), index=glathida_rgis.index)

train, test = create_train_test(glathida_rgis, rgi=None, full_shuffle=True, frac=.2, seed=None)
test_index = test.index
test_mask_bool = pd.Series(glathida_rgis.index.isin(test_index), index=glathida_rgis.index)

#test_mask_bool = ((glathida_rgis['RGI']==11) & (glathida_rgis['POINT_LON']<7.2))#<76)
#test_mask_bool = glathida_rgis.sample(1822).index
#random_bool_array = np.concatenate([np.full(1822, True), np.full(len(glathida_rgis) - 1822, False)])
#test_mask_bool = pd.Series(np.random.permutation(random_bool_array), index=glathida_rgis.index)
#test_mask_bool = ((glathida_rgis['RGI']==3) & (glathida_rgis['POINT_LAT']<76))#<76)

train_mask_bool[test_mask_bool] = False
#print(len(train_mask_bool), train_mask_bool.sum())

some_val_indexes = train_mask_bool[test_mask_bool==False].sample(1000).index
train_mask_bool[some_val_indexes] = False
#print(len(train_mask_bool), train_mask_bool.sum())

val_mask_bool[some_val_indexes] = True
#print(len(val_mask_bool), val_mask_bool.sum())


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
criterion = CFG.loss
optimizer = Adam(model.parameters(), lr=CFG.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=CFG.L2_penalty)

best_rmse = np.inf
best_weights = None


for epoch in range(CFG.epochs):
    model.train()
    optimizer.zero_grad(data)
    out = model(data) # NB out contiene TUTTI i nodi #, out_m, out_f
    loss3 = criterion(out[data.train_mask], data.y[data.train_mask])
    #loss1 = criterion(out_m[data.train_mask], data.y[data.train_mask])
    #loss2 = criterion(out_f[data.train_mask], data.y[data.train_mask])
    loss = loss3#loss1 + loss2 + 2 * loss3
    loss.backward()
    optimizer.step()

    rmse = mean_squared_error(out[data.train_mask].detach().cpu().numpy(),
                              data.y[data.train_mask].detach().cpu().numpy(), squared=False)

    model.eval()
    with torch.no_grad():
        out = model(data) #, out_m, out_f
        loss_val3 = criterion(out[data.val_mask], data.y[data.val_mask])
        #loss_val1 = criterion(out_m[data.val_mask], data.y[data.val_mask])
        #loss_val2 = criterion(out_f[data.val_mask], data.y[data.val_mask])
        loss_val = loss_val3 #loss_val1 + loss_val2 + 2 * loss_val3
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
out = best_model(data) #, out_m, out_f
y_preds = out[data.test_mask].detach().cpu().numpy().squeeze()

y_test_m = glathida_rgis[CFG.millan][test_mask_bool==True].to_numpy()
y_test_f = glathida_rgis[CFG.farinotti][test_mask_bool==True].to_numpy()

# Inference on a specific glacier
glacier_name_for_generation = get_random_glacier_rgiid(name='RGI60-07.00832', rgi=3, area=20, seed=None)
test_glacier = populate_glacier_with_metadata(glacier_name=glacier_name_for_generation,
                                              n=CFG.n_points_regression, seed=None, verbose=True)
test_glacier_rgi = glacier_name_for_generation[6:8]

# Create glacier graph
# Calculate pair-wise haversine distances between all points in a vectorized fashion
test_lon = np.array(test_glacier['lons'])
test_lat = np.array(test_glacier['lats'])
test_distances = haversine(test_lon, test_lat, test_lon[:, np.newaxis], test_lat[:, np.newaxis])
print(f'Test glacier matrix of pair-wise distances: {test_distances.shape}')
# Adjacency matrix
test_adj_matrix = np.where(test_distances < CFG.threshold, 1, 0)
# remove self loops (set diagonal to zero)
np.fill_diagonal(test_adj_matrix, 0)

# Edges
# find indexes corresponding to 1 in the adjacency matrix
test_edge_index = np.argwhere(test_adj_matrix==1) # (node_indexes, 2)
test_edge_index = test_edge_index.transpose() # (2, node_indexes)
print(f'Edge index vector: {test_edge_index.shape}')

test_edge_weight = test_distances[test_adj_matrix==1]
print(f'Check min and max should be > 0 and < threshold: {np.min(test_edge_weight)}, {np.max(test_edge_weight)}')
test_edge_weight = 1./(test_edge_weight ** 1) #np.zeros_like(edge_weight) #

# graph
test_data = Data(x=torch.tensor(test_glacier[CFG.features].to_numpy(), dtype=torch.float32),
            edge_index=torch.tensor(test_edge_index, dtype=torch.long),
            edge_weight=torch.tensor(test_edge_weight, dtype=torch.float32),
            y = None,
            #y=torch.tensor(test_glacier[CFG.target].to_numpy().reshape(-1, 1), dtype=torch.float32),
            m=torch.tensor(test_glacier[CFG.millan].to_numpy().reshape(-1, 1), dtype=torch.float32),
            f=torch.tensor(test_glacier[CFG.farinotti].to_numpy().reshape(-1, 1), dtype=torch.float32)
            )

best_model.to('cpu')
test_data = test_data.to('cpu')
test_out = best_model(test_data)
y_preds_glacier = test_out.detach().cpu().numpy().squeeze()


plot_fancy_ML_prediction = True
if plot_fancy_ML_prediction:
    fig, ax = plt.subplots(figsize=(8,6))

    # Begin to extract all necessary things to plot the result
    oggm_rgi_shp = glob(f"{args.oggm}rgi/RGIV62/{test_glacier_rgi}*/{test_glacier_rgi}*.shp")[0]
    oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp, engine='pyogrio')
    glacier_geometry = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == glacier_name_for_generation][
        'geometry'].item()
    glacier_area = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId'] == glacier_name_for_generation]['Area'].item()
    exterior_ring = glacier_geometry.exterior  # shapely.geometry.polygon.LinearRing
    glacier_nunataks_list = [nunatak for nunatak in glacier_geometry.interiors]

    swlat = test_glacier['lats'].min()
    swlon = test_glacier['lons'].min()
    nelat = test_glacier['lats'].max()
    nelon = test_glacier['lons'].max()
    deltalat = np.abs(swlat - nelat)
    deltalon = np.abs(swlon - nelon)
    eps = 5. / 3600
    focus_mosaic_tiles = create_glacier_tile_dem_mosaic(minx=swlon - (deltalon + eps),
                                                        miny=swlat - (deltalat + eps),
                                                        maxx=nelon + (deltalon + eps),
                                                        maxy=nelat + (deltalat + eps),
                                                        rgi=test_glacier_rgi, path_tandemx=args.mosaic)
    focus = focus_mosaic_tiles.squeeze()

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
    s_glathida = ax.scatter(x=glathida_rgis['POINT_LON'], y=glathida_rgis['POINT_LAT'], c=glathida_rgis['THICKNESS'],
                            cmap='jet', ec='grey', lw=0.5, s=20, vmin=vmin,vmax=vmax)

    cbar = plt.colorbar(s1, ax=ax)
    cbar.mappable.set_clim(vmin=vmin,vmax=vmax)
    cbar.set_label('Thickness (m)', labelpad=15, rotation=90, fontsize=14)
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

exit()

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

ax2.hist(y_test-y_preds, bins=np.arange(-1000, 1000, 10), label='GNN', color='lightblue', ec='blue', alpha=.4, zorder=2)
ax2.hist(y_test-y_test_m, bins=np.arange(-1000, 1000, 10), label='Millan', color='green', ec='green', alpha=.3, zorder=1)
ax2.hist(y_test-y_test_f, bins=np.arange(-1000, 1000, 10), label='Farinotti', color='red', ec='red', alpha=.3, zorder=3)

# text
text_ml = f'GNN\n$\\mu$ = {mu_ML:.1f}\nmed = {med_ML:.1f}\n$\\sigma$ = {std_ML:.1f}'
text_millan = f'Millan\n$\\mu$ = {mu_millan:.1f}\nmed = {med_millan:.1f}\n$\\sigma$ = {std_millan:.1f}'
text_farinotti = f'Farinotti\n$\\mu$ = {mu_farinotti:.1f}\nmed = {med_farinotti:.1f}\n$\\sigma$ = {std_farinotti:.1f}'
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

dataset_test = glathida_rgis[test_mask_bool]
print('Test dataset:', len(dataset_test))
test_glaciers_names = dataset_test['RGIId'].unique().tolist()
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

s1 = ax1.scatter(x=dataset_test['POINT_LON'], y=dataset_test['POINT_LAT'], s=10, c=y_test, cmap='Blues', label='Glathida')
s2 = ax2.scatter(x=dataset_test['POINT_LON'], y=dataset_test['POINT_LAT'], s=10, c=y_preds, cmap='Blues', label='GNN')
s3 = ax3.scatter(x=dataset_test['POINT_LON'], y=dataset_test['POINT_LAT'], s=10, c=y_test_m, cmap='Blues', label='Millan')
s4 = ax4.scatter(x=dataset_test['POINT_LON'], y=dataset_test['POINT_LAT'], s=10, c=y_test_f, cmap='Blues', label='Farinotti')
s5 = ax5.scatter(x=dataset_test['POINT_LON'], y=dataset_test['POINT_LAT'], s=10, c=y_test-y_preds, cmap='bwr', label='Glathida-GNN')
s6 = ax6.scatter(x=dataset_test['POINT_LON'], y=dataset_test['POINT_LAT'], s=10, c=y_test-y_test_f, cmap='bwr', label='Glathida-Farinotti')

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

