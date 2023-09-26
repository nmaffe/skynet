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

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import GraphConv


PATH_METADATA = '/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/'
file = 'TTT_final_grid_20.csv'

glathida_rgis = pd.read_csv(PATH_METADATA+file, low_memory=False)
print(f'Dataset: {len(glathida_rgis)} rows and', glathida_rgis['RGIId'].nunique(), 'glaciers.')
print(list(glathida_rgis))


def edge_0_1(lon1, lat1, lon2, lat2, threshold=3.0):

    dist_km = haversine(lon1, lat1, lon2, lat2) #km

    if (dist_km <= threshold): edge = 1
    else: edge = 0

    print(dist_km, edge)
    return edge

edge = edge_0_1(0, 0, 0, 0.1, threshold=5)
