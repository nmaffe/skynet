import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy import stats

PATH_IN = '/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_final.csv'
glathida = pd.read_csv(PATH_IN, low_memory=False)
glathida.describe().T
print(list(glathida))
