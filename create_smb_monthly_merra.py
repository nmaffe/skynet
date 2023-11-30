import os
import glob
import numpy as np
import xarray as xr
import rioxarray
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import geopandas as gpd

PATH_MERRA2_INPUT_FOLDER = '/media/nico/samsung_nvme/MERRA-2/'

#year = 2017
for year, month in product(range(2017, 2019), range(1,13)):

    print(f'Year: {year}, Month: {month}')

    file_int =  f'MERRA2_400.tavgM_2d_int_Nx.{year}{month:02}.nc4'
    file_glc =  f'MERRA2_400.tavgM_2d_glc_Nx.{year}{month:02}.nc4'

    # Import files and create Datasets
    ds_glc = rioxarray.open_rasterio(PATH_MERRA2_INPUT_FOLDER + f'{year}' + '/' + file_glc, nodata=1e15)
    ds_int = rioxarray.open_rasterio(PATH_MERRA2_INPUT_FOLDER + f'{year}' + '/' + file_int, nodata=np.nan)

    # Create DataArray and replace fill values with nans
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


    # Calculate SMB DataArray and create SMB Dataset
    # todo: should EVAP be added or subtracted ?
    SMB = ds_int['PRECCU'] + ds_int['PRECLS'] + ds_int['PRECSN'] - ds_int['EVAP'] - ds_glc['RUNOFF'].values
    ds_smb = xr.Dataset(coords=ds_int.coords)
    ds_smb['SMB'] = SMB
    #print('Dataset: ', ds_smb)

    # Plot
    plot = False
    if plot:
        fig, axs = plt.subplots(3,2, figsize=(10,6))
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
        im1 = ax1.imshow(ds_int['PRECCU'].values.squeeze(), cmap='bwr')
        im2 = ax2.imshow(ds_int['PRECLS'].values.squeeze(), cmap='bwr')
        im3 = ax3.imshow(ds_int['PRECSN'].values.squeeze(), cmap='bwr')
        im4 = ax4.imshow(ds_int['EVAP'].values.squeeze(), cmap='bwr')
        im5 = ax5.imshow(ds_glc['RUNOFF'].values.squeeze(), cmap='bwr')
        absmax = max(abs(np.min(ds_smb['SMB'])), abs(np.max(ds_smb['SMB'])))
        im6 = ax6.imshow(ds_smb['SMB'].values.squeeze(), vmin=-absmax, vmax=absmax, cmap='bwr')

        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04)
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.03, pad=0.04)
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.03, pad=0.04)
        cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.03, pad=0.04)
        cbar6 = fig.colorbar(im6, ax=ax6, fraction=0.03, pad=0.04)

        cbar1.set_label(ds_int['PRECCU'].attrs['long_name']+'\n'+ds_int['PRECCU'].attrs['units'], rotation=90, labelpad=15)
        cbar2.set_label(ds_int['PRECLS'].attrs['long_name']+'\n'+ds_int['PRECLS'].attrs['units'], rotation=90, labelpad=15)
        cbar3.set_label(ds_int['PRECSN'].attrs['long_name']+'\n'+ds_int['PRECSN'].attrs['units'], rotation=90, labelpad=15)
        cbar4.set_label(ds_int['EVAP'].attrs['long_name']+'\n'+ds_int['EVAP'].attrs['units'], rotation=90, labelpad=15)
        cbar5.set_label(ds_glc['RUNOFF'].attrs['long_name']+'\n'+ds_glc['RUNOFF'].attrs['units'], rotation=90, labelpad=15)
        cbar6.set_label('SBM'+'\n'+'kg m-2 s-1', rotation=90, labelpad=15)

        plt.tight_layout()
        plt.show()

        # ds_glc['RUNOFF'].plot(cmap='bwr')
        # plt.show()

    # Save surface mass balance Dataset to disk
    save = True
    if save:
        ds_smb.to_netcdf(f"{PATH_MERRA2_INPUT_FOLDER}/{year}/{file_int.replace('int', 'smb')}")
        print(f"saved.")


