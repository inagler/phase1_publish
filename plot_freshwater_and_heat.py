#!/usr/bin/env python3
# inagler 19/01/25

import os
import glob
import dask
import numpy as np
import xarray as xr
import cftime
import pop_tools
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib as mpl
import pandas as p

# Helper function to standardize time format
def standardize_time(ds):
    decoded = xr.decode_cf(ds, use_cftime=True)
    if isinstance(decoded.time.values[0], cftime._cftime.DatetimeNoLeap):
        time_as_datetime64 = np.array([pd.Timestamp(str(dt)).to_datetime64() for dt in decoded.time.values])
        ds['time'] = xr.DataArray(time_as_datetime64, dims='time')
    return ds

# Define region-specific masks
grid_name = 'POP_gx1v7'
region_defs = {
    'SubpolarAtlantic': [
        {'match': {'REGION_MASK': [6]}, 'bounds': {'TLAT': [50.0, 65.0], 'TLONG': [200.0, 360.0]}}
    ],
    'LabradorSea': [
        {'match': {'REGION_MASK': [8]}, 'bounds': {'TLAT': [50.0, 65.0], 'TLONG': [260.0, 360.0]}}
    ]
}

# Create mask for selected regions
mask = pop_tools.region_mask_3d(grid_name, region_defs=region_defs, mask_name='North Atlantic')
mask = mask.sum('region')

# Define path and files pattern for temperature and salinity data
path = '/Data/skd/scratch/innag3580/comp/averages/'
temp_files_pattern = 'TEMP_*.nc'
salt_files_pattern = 'SALT_*.nc'

# Load temperature data, standardize time, and concatenate along the time dimension using Dask
temp_file_list = sorted(glob.glob(os.path.join(path, temp_files_pattern)))
ds_temp = xr.open_mfdataset(temp_file_list, chunks={'time': 120}, preprocess=standardise_time)
ds_temp['TEMP'] = ds_temp['TEMP'].astype('float32')
ds_temp = ds_temp.where(mask == 1).resample(time='1Y').mean()

# Load salinity data, standardize time, and concatenate along the time dimension using Dask
salt_file_list = sorted(glob.glob(os.path.join(path, salt_files_pattern)))
ds_salt = xr.open_mfdataset(salt_file_list, chunks={'time': 120}, preprocess=standardise_time)
ds_salt['SALT'] = ds_salt['SALT'].astype('float32')
ds_salt = ds_salt.where(mask == 1).resample(time='1Y').mean()

# Define physical constants
rho_sw = 1026  # density of seawater in kg/m^3
cp_sw = 3990  # specific heat of seawater in J/(kg·K)

# Convert units to meters and square meters
ds_salt['dz'] = ds_temp.dz * 1e-2
ds_temp['UAREA'] = ds_temp.UAREA * 1e-4

# Define reference salinity
S_ref = 35  # PSU

# Convert units to meters and square meters
ds_salt['dz'] = ds_salt.dz * 1e-2
ds_salt['UAREA'] = ds_salt.UAREA * 1e-4

### COMPUTATION ###

# Compute heat content
heat_content = rho_sw * cp_sw * (ds_temp.dz * ds_temp.UAREA * ds_temp.TEMP).sum(dim=['nlat', 'nlon', 'z_t'])

# Freshwater content computation (add this line)
freshwater_content = ((S_ref - ds_salt.SALT) / S_ref * ds_salt.dz * ds_salt.UAREA).sum(dim=['nlat', 'nlon', 'z_t'])

### OUTPUT ###

# Plot heat content time series
fontsize = 14
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(heat_content.time, heat_content / 1e23, color='b', label='Heat Content')
ax.set_xlabel('Year', fontsize=fontsize)
ax.set_ylabel('Heat Content ($10^{23}$ J)', fontsize=fontsize)
ax.set_title('Heat Content Time Series in the Subpolar North Atlantic', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=fontsize)
fig.tight_layout()
plt.savefig('Figure_heat_content.png', bbox_inches='tight', dpi=300)

# Plot freshwater content time series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freshwater_content.time, freshwater_content / 1e9, color='b', label='Freshwater Content')
ax.set_xlabel('Year', fontsize=fontsize)
ax.set_ylabel('Freshwater Content ($10^9$ m³)', fontsize=fontsize)
ax.set_title('Total Freshwater Content Time Series in the Subpolar North Atlantic', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
plt.savefig('Figure_freshwater_content.png', bbox_inches='tight', dpi=300)

# Plot both heat and freshwater content time series
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(heat_content.time, heat_content / 1e23, color='b', label='Heat Content')
ax1.set_xlabel('Year', fontsize=fontsize)
ax1.set_ylabel('Heat Content ($10^{23}$ J)', fontsize=fontsize)
ax1.tick_params(axis='both', which='major', labelsize=fontsize)
ax2 = ax1.twinx()
ax2.plot(freshwater_content.time, freshwater_content / 1e9, color='r', label='Freshwater Content', alpha=0.6)
ax2.set_ylabel('Freshwater Content ($10^9$ m³)', fontsize=fontsize)
ax2.tick_params(axis='both', which='major', labelsize=fontsize)
fig.suptitle('Heat and Freshwater Content Time Series in the Subpolar North Atlantic', fontsize=fontsize)
ax1.legend(loc='upper left', fontsize=fontsize)
ax2.legend(loc='upper right', fontsize=fontsize)
ax1.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Figure_freshwater_and_heat.png', bbox_inches='tight', dpi=300)