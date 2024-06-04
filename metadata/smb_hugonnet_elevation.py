import os, sys
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import mode, gaussian_kde
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import geopandas as gpd
import oggm
from oggm import utils
from utils_metadata import haversine

utils.get_rgi_dir(version='62')  # setup oggm version
mbdf = utils.get_geodetic_mb_dataframe() # note that this takes 1.3s. I should avoid opening this for each glacier
mbdf = mbdf.loc[mbdf['period']=='2000-01-01_2020-01-01']

rgi = 3

mbdf_rgi = mbdf.loc[mbdf['reg'] == rgi]
oggm_rgi_shp = utils.get_rgi_region_file(f"{rgi:02d}", version='62')
oggm_rgi_glaciers = gpd.read_file(oggm_rgi_shp)
if len(oggm_rgi_glaciers)>20000:
    rgiids = oggm_rgi_glaciers['RGIId'].sample(20000).unique()
else: rgiids = oggm_rgi_glaciers['RGIId'].unique()
print(len(mbdf_rgi), len(oggm_rgi_glaciers), len(rgiids))

zmins, zmeds, zmaxs, smbs, lats, lons = [], [], [], [], [], []

for i, rgiid in enumerate(rgiids):
    zmin = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==rgiid, 'Zmin'].item()
    zmed = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==rgiid, 'Zmed'].item()
    zmax = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==rgiid, 'Zmax'].item()
    lat = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==rgiid, 'CenLat'].item()
    lon = oggm_rgi_glaciers.loc[oggm_rgi_glaciers['RGIId']==rgiid, 'CenLon'].item()

    if rgi==19:
        if (zmin == -999 or zmax == -999):
            continue
        else:
            zmed = 0.5 * (zmin + zmax)
    else:
        if (zmin == -999 or zmed == -999 or zmax == -999):
            continue

    try:
        glacier_dmdtda = mbdf_rgi.at[rgiid, 'dmdtda']
    except:
        continue

    print(i,'/',len(rgiids), rgiid, zmin, zmed, zmax, glacier_dmdtda)

    zmins.append(zmin)
    zmeds.append(zmed)
    zmaxs.append(zmax)
    lats.append(lat)
    lons.append(lon)
    smbs.append(glacier_dmdtda)

assert len(zmins) == len(zmeds) == len(zmaxs) == len(lats) == len(lons) == len(smbs), 'Something wrong in sizes.'

zmins, zmaxs, zmeds = np.array(zmins), np.array(zmaxs), np.array(zmeds)
lats, lons = np.array(lats), np.array(lons)
smbs = np.array(smbs)

mode_smb = mode(smbs)[0]
print('smb mode:', mode_smb)

zmins = zmins[smbs != mode_smb]
zmaxs = zmaxs[smbs != mode_smb]
zmeds = zmeds[smbs != mode_smb]
lats = lats[smbs != mode_smb]
lons = lons[smbs != mode_smb]
smbs = smbs[smbs != mode_smb]

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.scatter(x=zmins, y=smbs, c='r', s=5)
ax2.scatter(x=zmeds, y=smbs, c='g', s=5)
ax3.scatter(x=zmaxs, y=smbs, c='b', s=5)
plt.show()

#fig, ax = plt.subplots()
#s = ax.scatter(x=lons, y=lats, c=smbs)
#plt.colorbar(s)
#plt.show()

# Create a mask for the diagonal elements: true on diagonal, false off diagonal
mask_self = np.eye(len(smbs), dtype=bool)

# create matrix of relative distances
# meshgrid for vectorized haversine computation
lat1, lat2 = np.meshgrid(lats, lats)
lon1, lon2 = np.meshgrid(lons, lons)
# dists[i,j] represent the distance between point i and point j from the glacier positions (lons, lats)
dists = 1000 * haversine(lon1, lat1, lon2, lat2) # m
#dists = np.where(dists>2000, 0.0, dists)
dists_masked_self = np.ma.array(dists, mask=mask_self)
dists_inv = (1. / dists_masked_self).filled(0.0) # Zero on diagonal
#print(dists_inv)

#print(lons[0], lats[0])
#print(lons[1], lats[1])
#print(dists[0,1], dists[1,0], haversine(lons[0], lats[0], lons[1], lats[1]))

#fig, ax = plt.subplots()
#ax.scatter(x=lons[0], y=lats[0], c='r', s=30, zorder=1)
#s2 = ax.scatter(x=lons[1:], y=lats[1:], c=dists_inv[0, 1:], s=3)
#cbar = plt.colorbar(s2)
#plt.show()

# important decision: decide that is x: hmin, hmed or hmax
y = np.array(smbs)
h = np.array(zmins)

dh = 1.0 * h[:, np.newaxis] - h # anti-symmetric: (i,j) = -(j,i)
dy = 1.0 * y[:, np.newaxis] - y # anti-symmetric: (i,j) = -(j,i)
#plt.hist(dy.ravel(), bins=100, alpha=0.5)
#plt.show()

dy_masked = np.ma.array(dy, mask=mask_self)  # masks matrix on diagonal
dh_masked = np.ma.array(dh, mask=mask_self)  # masks matrix on diagonal

# Perform the division without considering diagonal elements. This matrix is symmetrical
ratios = (dy_masked / dh_masked) # * np.sum(dists, axis=0)
#plt.hist(np.mean(ratios, axis=0).ravel(), bins=100, alpha=0.5)
#plt.show()
# Normalize the weights to make them dimensionless
weighted_ratios = np.average(ratios, axis=0, weights=dists_inv)
print(weighted_ratios.shape)
print(np.min(dy), np.max(dy))
print(np.min(dh), np.max(dh))
print(np.min(ratios), np.max(ratios), np.mean(ratios))
print(np.min(weighted_ratios), np.max(weighted_ratios), np.mean(weighted_ratios))
print(f"m: {np.mean(weighted_ratios)}, std: {np.std(weighted_ratios)}")
fig, (ax0, ax) = plt.subplots(1,2)
hist = ax0.hist(weighted_ratios, bins=100)
s = ax.scatter(x=lons, y=lats, c=weighted_ratios, cmap='bwr', vmin=-0.0006, vmax=0.0006)
plt.colorbar(s)
plt.show()

# Get rid of points further away from 3 sigma from the mean
mask_3s = np.abs(weighted_ratios - np.mean(weighted_ratios)) <= 3 * np.std(weighted_ratios)
weighted_ratios = weighted_ratios[mask_3s]

# Calculate the mean m
m = np.mean(weighted_ratios)

# Calculate the mean q
qs = y - m * h
q = np.mean(qs)

print(f"After outlier removal m: {m}, std: {np.std(weighted_ratios):.5f}, stdperc {np.std(weighted_ratios)*100/m:.1f}")
print(f"q: {q}, std: {np.std(qs):.5f}, stdperc {np.std(qs)*100/q:.1f}")

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(weighted_ratios, bins=300)
ax2.hist(qs, bins=300)
plt.show()


# Rialcolare m locale vincolando q a una costante regionale. PAZZIA ?
# m_glaciers = []
# for i, (zmin, zmax, zmed, smb) in enumerate(zip(zmins, zmaxs, zmeds, smbs)):
#     print(i, '/', len(zmins))
#
#     m = (smb - q)/max(1, zmin)
#     m_glaciers.append(m)
#
# m_glaciers = np.array(m_glaciers)
# mask_3s = np.abs(m_glaciers - np.mean(m_glaciers)) <= 3 * np.std(m_glaciers)
# m_glaciers = m_glaciers[mask_3s]
#
# fig, ax = plt.subplots()
# hist = ax.hist(m_glaciers, bins=1000)
# plt.show()


