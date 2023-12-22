import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy import stats

PATH_METADATA = '/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_final.csv'
glathida = pd.read_csv(PATH_METADATA, low_memory=False)
print(list(glathida))

""" A. Work on the dataset """

# A.1 Add features
glathida['v'] = np.sqrt(glathida['vx']**2 + glathida['vy']**2)
glathida['slope'] = np.sqrt(glathida['slope_lat']**2 + glathida['slope_lon']**2)
glathida['elevation_from_zmin'] = glathida['elevation_astergdem'] - glathida['Zmin']

# A.2 Select only regions of interest, remove old (-er than 2005) measurements
# and erroneous data (if DATA_FLAG is not nan)
rgis = [3,7,8,11]

cond = ((glathida['RGI'].isin(rgis)) & (glathida['SURVEY_DATE'] > 20050000)
        & (glathida['DATA_FLAG'].isna()) & (glathida['THICKNESS']>=0)) #& (glathida['ith_m']>0)

glathida_rgis = glathida[cond]
print(f'Original columns: {list(glathida_rgis)} \n')

# A.3 Keep only these columns
cols = ['RGI', 'RGIId', 'POINT_LAT', 'POINT_LON', 'THICKNESS', 'Area', 'elevation_astergdem',
        'slope_lat', 'slope_lon', 'vx', 'vy', 'dist_from_border_km', 'v', 'slope',
       'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'elevation_from_zmin', 'ith_m', 'ith_f']

glathida_rgis = glathida_rgis[cols]
#print(f'We keep only the following columns: \n {list(glathida_rgis)} \n{len(glathida_rgis)} rows')

# A.4 Remove nans
glathida_rgis = glathida_rgis.dropna(subset=cols)
#print(glathida_rgis.isna().sum())
#print(f'After having removed nans we have {len(glathida_rgis)} rows')
#print(glathida_rgis['RGI'].value_counts())

""" B. Grid the dataset """
# We loop over all unique glacier ids; for each unique glacier we grid every feature.
ids_rgiid = glathida_rgis['RGIId'].unique().tolist()
print(f'We have {len(ids_rgiid)} unique glaciers')

#todo: CONTINUE TO IMPLEMENT FROM .IPYNB