import os
import glob
import numpy as np
import xarray as xr
import rioxarray
import matplotlib
import matplotlib.pyplot as plt
from oggm import utils
import geopandas as gpd

PATH_MERRA2_FOLDER = '/media/nico/samsung_nvme/MERRA-2/'
year = 2017
file_glc = PATH_MERRA2_FOLDER + str(year) + '/' + 'MERRA2_400.tavgM_2d_glc_Nx.201701.nc4'
file_int = PATH_MERRA2_FOLDER + str(year) + '/' + 'MERRA2_400.tavgM_2d_int_Nx.201701.nc4'

ds_glc = rioxarray.open_rasterio(file_glc, nodata=1e15)
ds_int = rioxarray.open_rasterio(file_int, nodata=np.nan)

ds_int['PRECCU'] = ds_int['PRECCU'].where(ds_int['PRECCU'] != 1e15)
ds_int['PRECLS'] = ds_int['PRECLS'].where(ds_int['PRECLS'] != 1e15)
ds_int['PRECSN'] = ds_int['PRECSN'].where(ds_int['PRECSN'] != 1e15)
ds_int['EVAP'] = ds_int['EVAP'].where(ds_int['EVAP'] != 1e15)
ds_glc['RUNOFF'] = ds_glc['RUNOFF'].where(ds_glc['RUNOFF'] != 1e15)

ds_int['PRECCU'].attrs['_FillValue'] = np.nan
ds_int['PRECLS'].attrs['_FillValue'] = np.nan
ds_int['PRECSN'].attrs['_FillValue'] = np.nan
ds_int['EVAP'].attrs['_FillValue'] = np.nan
ds_glc['RUNOFF'].attrs['_FillValue'] = np.nan

# Calculate SMB
# todo: should EVAP be added or subtracted ?
SMB = ds_int['PRECCU'] + ds_int['PRECLS'] + ds_int['PRECSN'] - ds_int['EVAP'] - ds_glc['RUNOFF']

# Create SMB Dataset with the SMB DataArray
ds_smb = xr.Dataset(coords=ds_int.coords)
ds_smb['SMB'] = SMB

print(ds_int['PRECCU'].isnull().sum())
print(ds_int['PRECLS'].isnull().sum())
print(ds_int['PRECSN'].isnull().sum())
print(ds_int['EVAP'].isnull().sum())
print(ds_glc['RUNOFF'].isnull().sum())
print(float(ds_smb['SMB'].isnull().sum()))


# Plot
ds_glc['RUNOFF'].plot(cmap='bwr')
plt.show()