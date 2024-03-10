import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal

"""
This program imports the generated metadata dataset from create_metadata.py and:
1. Processing:
    - Remove old measurements
    - Remove possible bad data
    - Retain only some final features and remove any nan.
2. Gridding:
    - gridding is done computing the mean of each feature in all pixels that form the grid, which is specified by
    the parameter --nbins_grid_latlon (default=20)
    
The processed and gridded dataframe is finally saved.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input_metadata_file', type=str,
                    default="/home/nico/PycharmProjects/skynet/Extra_Data/glathida/glathida-3.1.0/glathida-3.1.0/data/metadata.csv",
                    help="Input metadata file to be gridded")
parser.add_argument('--tmin', type=int, default=20050000, help="Keep only measurements after this year.")
parser.add_argument('--hmin', type=float, default=0.0, help="Keep only measurements with thickness greater than this.")
parser.add_argument('--method_grid', type=str, default='mean', help="Supported options: mean, median")
parser.add_argument('--nbins_grid_latlon', type=int, default=20, help="How many bins in the lat/lon directions")
parser.add_argument('--save', type=bool, default=False, help="Save final dataset or not.")

args = parser.parse_args()


""" Import ungridded dataset """
glathida = pd.read_csv(args.input_metadata_file, low_memory=False)

""" A. Work on the dataset """
# A.1 Remove old (-er than 2005) measurements and erroneous data (if DATA_FLAG is not nan)
cond = ((glathida['SURVEY_DATE'] > args.tmin) & (glathida['DATA_FLAG'].isna()) & (glathida['THICKNESS']>=args.hmin))

glathida = glathida[cond]
print(f'Original columns: {list(glathida)} \n')

# A.2 Keep only these columns
cols = ['RGI', 'RGIId', 'POINT_LAT', 'POINT_LON', 'THICKNESS', 'Area', 'elevation',
        'slope_lat', 'slope_lon', 'vx', 'vy', 'dist_from_border_km', 'dist_from_border_km_geom',
       'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'ith_m', 'ith_f',
        'slope_lon_gf50', 'slope_lat_gf50', 'slope_lon_gf100', 'slope_lat_gf100', 'slope_lon_gf150', 'slope_lat_gf150',
        'slope_lon_gf300', 'slope_lat_gf300', 'slope_lon_gf450', 'slope_lat_gf450', 'Form', 'Aspect', 'TermType', 'vx_gf50', 'vx_gf100', 'vx_gf150',
        'vx_gf300', 'vy_gf50', 'vy_gf100', 'vy_gf150', 'vy_gf300', 'dvx_dx', 'dvx_dy', 'dvy_dx', 'dvy_dy',
        'curv_50', 'curv_300', 'aspect_50', 'aspect_300']

glathida = glathida[cols]
print(f'We keep only the following columns: \n {list(glathida)} \n{len(glathida)} rows')

# A.3 Remove nans
glathida = glathida.dropna(subset=cols)
#print(glathida.isna().sum())
#print(f'After having removed nans we have {len(glathida)} rows')
#print(glathida['RGI'].value_counts())

""" B. Grid the dataset """
# We loop over all unique glacier ids; for each unique glacier we grid every feature.
print(f"Begin gridding.")
rgi_ids = glathida['RGIId'].unique().tolist()
print(f'We have {len(rgi_ids)} unique glaciers and {len(glathida)} rows')

glathida_gridded = pd.DataFrame(columns=glathida.columns)

features_to_grid = ['THICKNESS', 'Area', 'elevation',
        'slope_lat', 'slope_lon', 'vx', 'vy', 'dist_from_border_km', 'dist_from_border_km_geom',
       'Zmin', 'Zmax', 'Zmed', 'Slope', 'Lmax', 'ith_m', 'ith_f',
        'slope_lon_gf50', 'slope_lat_gf50', 'slope_lon_gf100', 'slope_lat_gf100', 'slope_lon_gf150', 'slope_lat_gf150',
        'slope_lon_gf300', 'slope_lat_gf300', 'slope_lon_gf450', 'slope_lat_gf450', 'vx_gf50', 'vx_gf100', 'vx_gf150',
        'vx_gf300', 'vy_gf50', 'vy_gf100', 'vy_gf150', 'vy_gf300', 'dvx_dx', 'dvx_dy', 'dvy_dx', 'dvy_dy',
                    'curv_50', 'curv_300', 'aspect_50', 'aspect_300']

list_num_measurements_before_grid = []
list_num_measurements_after_grid = []

# loop over unique glaciers
for n, rgiid in enumerate(rgi_ids):

    glathida_id = glathida.loc[glathida['RGIId'] == rgiid]
    glathida_id_grid = pd.DataFrame(columns=glathida_id.columns)

    lons = glathida_id['POINT_LON'].to_numpy()
    lats = glathida_id['POINT_LAT'].to_numpy()
    rgi = glathida_id['RGI'].iloc[0]
    form = glathida_id['Form'].iloc[0]
    aspect = glathida_id['Aspect'].iloc[0]
    termtype = glathida_id['TermType'].iloc[0]

    # make same checks
    if not glathida_id['Area'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique Area.")
    if not glathida_id['RGI'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique RGI.")
    if not glathida_id['RGIId'].nunique() == 1: raise ValueError(f"Glacier {rgiid} should have only 1 unique RGIId.")

    print(f'{n}/{len(rgi_ids)}, {rgiid}, Tot. meas to be gridded: {len(glathida_id)}')

    list_num_measurements_before_grid.append(len(glathida_id))

    # if only one measurement, append that line as is
    if (len(glathida_id) == 1):
        glathida_gridded = pd.concat([glathida_gridded, glathida_id], ignore_index=True)
        list_num_measurements_after_grid.append(len(glathida_id))
        continue

    # if more than one measurement, calcule the rectangular domain for gridding
    min_lat, max_lat = np.min(lats), np.max(lats)
    min_lon, max_lon = np.min(lons), np.max(lons)
    eps = 1.e-4
    binsx = np.linspace(min_lon-eps, max_lon+eps, num=args.nbins_grid_latlon)
    binsy = np.linspace(min_lat-eps, max_lat+eps, num=args.nbins_grid_latlon)
    assert len(binsx) == len(binsy) == args.nbins_grid_latlon, "Number of bins unexpected."

    # loop over each feature and grid
    for feature in features_to_grid:

        feature_array = glathida_id[feature].to_numpy()
        if np.isnan(feature_array).any(): raise ValueError('Watch out, nan in feature vector')

        if args.method_grid == 'mean':
            statistic = np.nanmean
        elif args.method_grid == 'median':
            statistic = np.nanmedian
        else: raise ValueError("method not supported.")

        # grid the feature
        H, xedges, yedges, binnumber = stats.binned_statistic_2d(x=lons, y=lats, values=feature_array,
                                                                 statistic=statistic, bins=[binsx, binsy])

        # calculate the latutide and longitude of the grid
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        non_nan_mask = ~np.isnan(H)
        x_indices, y_indices = np.where(non_nan_mask)
        xs = xcenters[x_indices]
        ys = ycenters[y_indices]
        zs = H[non_nan_mask]

        # check how many values we have produced
        new_gl_nmeas = np.count_nonzero(~np.isnan(H))
        # print(feature, new_gl_nmeas, H.shape, xedges.shape, xs.shape, ys.shape, zs.shape)

        # Fill gridded feature
        glathida_id_grid['POINT_LON'] = xs  # unnecessarily overwriting each loop
        glathida_id_grid['POINT_LAT'] = ys  # unnecessarily overwriting each loop
        glathida_id_grid[feature] = zs

        # plot
        ifplot = False
        if (feature == 'vx' and ifplot):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

            s1 = ax1.scatter(x=lons, y=lats, c=feature_array, s=1)
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

    # add these two features as well
    glathida_id_grid['RGI'] = np.repeat(rgi, len(glathida_id_grid))
    glathida_id_grid['RGIId'] = np.repeat(rgiid, len(glathida_id_grid))
    glathida_id_grid['Form'] = np.repeat(form, len(glathida_id_grid))
    glathida_id_grid['TermType'] = np.repeat(termtype, len(glathida_id_grid))
    glathida_id_grid['Aspect'] = np.repeat(aspect, len(glathida_id_grid))

    # append glacier gridded dataset to main gridded dataset
    glathida_gridded = pd.concat([glathida_gridded, glathida_id_grid], ignore_index=True)

    list_num_measurements_after_grid.append(len(glathida_id_grid))

print(f'Finished. No. original measurements {len(glathida)} down to {len(glathida_gridded)}.')

if args.save:
    filename_out = args.input_metadata_file.replace('.csv', f'_hmineq{args.hmin}_tmin{args.tmin}_{args.method_grid}_grid_{args.nbins_grid_latlon}.csv')
    glathida_gridded.to_csv(filename_out, index=False)
    print(f"Gridded dataframe saved: {filename_out}.")


