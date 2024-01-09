import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy import stats

PATH_METADATA_INPUT = '/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/TTT_final.csv'
glathida = pd.read_csv(PATH_METADATA_INPUT, low_memory=False)
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
print(f'We keep only the following columns: \n {list(glathida_rgis)} \n{len(glathida_rgis)} rows')

# A.4 Remove nans
glathida_rgis = glathida_rgis.dropna(subset=cols)
#print(glathida_rgis.isna().sum())
#print(f'After having removed nans we have {len(glathida_rgis)} rows')
#print(glathida_rgis['RGI'].value_counts())

""" B. Grid the dataset """
# We loop over all unique glacier ids; for each unique glacier we grid every feature.
print(f"Now we grid. \n")
ids_rgiid = glathida_rgis['RGIId'].unique().tolist()
print(f'We have {len(ids_rgiid)} unique glaciers')

glathida_rgis_gridded = pd.DataFrame(columns=glathida_rgis.columns)

features_to_grid = ['THICKNESS', 'Area', 'slope_lat', 'slope_lon', 'elevation_astergdem', 'vx', 'vy',
                    'dist_from_border_km', 'v', 'slope', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax',
                    'elevation_from_zmin', 'ith_m', 'ith_f']

num_grid_lon_lat = 20
n_measurements = []
n_measurements_new = []

# loop over unique glaciers
for n, idrgi in enumerate(ids_rgiid):

    glathida_id = glathida_rgis.loc[glathida_rgis['RGIId'] == idrgi]
    glathida_id_grid = pd.DataFrame(columns=glathida_id.columns)

    lons = glathida_id['POINT_LON'].to_numpy()
    lats = glathida_id['POINT_LAT'].to_numpy()
    gl_area = glathida_id['Area'].mean()
    gl_nmeas = len(glathida_id)
    gl_density = gl_nmeas / gl_area  # #/km2

    # control
    if (glathida_id['Area'].std() > 1.):
        print(glathida_id['Area'].std())
        input('PROBLEM')

    print(f'{n}/{len(ids_rgiid)}, {idrgi}, Tot. meas: {gl_nmeas}')

    n_measurements.append(gl_nmeas)

    # if only one measurement, append that line
    if (gl_nmeas == 1):
        glathida_rgis_gridded = glathida_rgis_gridded.append(glathida_id) #todo: deprecated use .concat instead
        n_measurements_new.append(gl_nmeas)
        continue

    # if more than one measurement, calcule the rectangular domain for gridding
    min_lat, max_lat = np.min(lats), np.max(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)
    binsx = np.arange(min_lon-1e-4, max_lon+1e-4, (max_lon-min_lon)/num_grid_lon_lat)
    binsy = np.arange(min_lat-1e-4, max_lat+1e-4, (max_lat-min_lat)/num_grid_lon_lat)

    # loop over features
    for feature in features_to_grid:

        vtest = glathida_id[feature].to_numpy()
        is_nan = np.isnan(np.sum(vtest))
        if is_nan: print('Watch out, nan in feature vector')

        # grid the feature
        H, xedges, yedges, binnumber = stats.binned_statistic_2d(x=lons, y=lats, values=vtest,
                                                                 statistic=np.nanmean,
                                                                 bins=[binsx, binsy])

        # check how many values we have produced
        new_gl_nmeas = np.count_nonzero(~np.isnan(H))

        # calculate the latutide and longitude of the grid
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        non_nan_mask = ~np.isnan(H)
        x_indices, y_indices = np.where(non_nan_mask)
        xs = xcenters[x_indices]
        ys = ycenters[y_indices]
        zs = H[non_nan_mask]

        # print(feature, new_gl_nmeas, H.shape, xedges.shape, xs.shape, ys.shape, zs.shape)

        # Fill gridded dataset
        rgi = glathida_id['RGI'].iloc[0]
        rgiid = glathida_id['RGIId'].iloc[0]
        assert rgiid==idrgi, "These should be the same"

        glathida_id_grid['POINT_LON'] = xs
        glathida_id_grid['POINT_LAT'] = ys
        glathida_id_grid[feature] = zs
        glathida_id_grid['RGI'] = np.repeat(rgi, len(zs))  # check
        glathida_id_grid['RGIId'] = np.repeat(rgiid, len(zs))  # check

        # plot
        ifplot = False
        if (feature == 'ith_m' and ifplot):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            s1 = ax1.scatter(x=lons, y=lats, c=vtest, s=1)
            cbar1 = plt.colorbar(s1, ax=ax1, alpha=1)
            cbar1.set_label(feature, labelpad=15, rotation=270)

            s2 = ax2.scatter(x=xs, y=ys, c=zs, s=1)
            cbar2 = plt.colorbar(s2, ax=ax2, alpha=1)
            cbar2.set_label(feature, labelpad=15, rotation=270)

            for x_edge in xedges:
                ax1.axvline(x_edge, color='gray', linestyle='--', linewidth=0.1)
                ax2.axvline(x_edge, color='gray', linestyle='--', linewidth=0.1)
            for y_edge in yedges:
                ax1.axhline(y_edge, color='gray', linestyle='--', linewidth=0.1)
                ax2.axhline(y_edge, color='gray', linestyle='--', linewidth=0.1)
            plt.show()

    # append glacier gridded dataset to main gridded dataset
    glathida_rgis_gridded = glathida_rgis_gridded.append(glathida_id_grid) #todo: deprecated use .concat instead

    n_measurements_new.append(len(glathida_id_grid))

ifsave = False
if ifsave:
    glathida_rgis_gridded.to_csv(PATH_METADATA_INPUT.replace('TTT_final.csv',
                                                 f'TTT_final_grid_{num_grid_lon_lat}_new.csv'), index=False)

print(f'Finished. No. original measurements {len(glathida_rgis)} down to {len(glathida_rgis_gridded)}.')
