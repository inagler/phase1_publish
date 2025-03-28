#!/usr/bin/env python3
# inagler 16/02/25

import os
import glob
import numpy as np
import xarray as xr
import csv
import pop_tools
import gsw

path = '/Data/skd/scratch/innag3580/comp/composites/'

# 'aa_hist', 'ghg_hist', 'early_hist', 'late_hist', 'above'
period = 'above'

if period == 'ghg_hist':
    ds_temp = xr.open_dataset(path + 'Below_combined_TEMP_3.0_40_20.nc')
    ds_salt = xr.open_dataset(path + 'Below_combined_SALT_3.0_40_20.nc')
    ds_vvel = xr.open_dataset(path + 'Below_combined_VVEL_3.0_40_20.nc')
elif perod == 'above':
    ds_temp = xr.open_dataset(path + 'Above_combined_TEMP_3.0_40_20.nc')
    ds_salt = xr.open_dataset(path + 'Above_combined_SALT_3.0_40_20.nc')
    ds_vvel = xr.open_dataset(path + 'Above_combined_VVEL_3.0_40_20.nc')
    
    ds_temp = ds_temp.isel(time=slice(0,3))
    ds_salt = ds_salt.isel(time=slice(0,3))
    ds_vvel = ds_vvel.isel(time=slice(0,3))
    

CT = gsw.conversions.CT_from_pt(ds_salt['SALT'], ds_temp['TEMP'])
sigma2 = gsw.density.sigma2(ds_salt['SALT'], CT)
sigma2 = xr.DataArray(sigma2, name='DENS2', dims=ds_temp['TEMP'].dims, coords=ds_temp['TEMP'].coords)

ds_dens = ds_temp
ds_dens = ds_dens.drop_vars('TEMP')
ds_dens['DENS2'] = sigma2

numbers = np.array([29.70, 31.50, 33.15, 34.75, 35.80, 36.38, 36.70, 36.89, 37.06, 37.13, 37.30])
intervals = np.diff(numbers) / 4
result = np.concatenate([np.arange(numbers[i], numbers[i+1], intervals[i]) for i in range(len(intervals))])
density_bins = np.append(result, numbers[-1])

print('initialised')

def calculate_smoc(ds_vvel, ds_dens, density_bins):
    
    cell_thickness = ds_vvel['dz']
    cell_width = ds_vvel['DXU']
    
    overturning_dataarray = xr.DataArray(np.zeros((len(ds_vvel.time), ds_vvel.dims['nlat'], len(density_bins) - 1), dtype=np.float32),
                                         dims=["time", "nlat", "density_bin"], 
                                         coords={"time": ds_vvel.time, "nlat": ds_vvel.nlat, "density_bin": density_bins[:-1]})
    
    for time_step in range(len(ds_vvel.time)):
        # compute meridional flow rate for each specified latitude
        velocity = ds_vvel['VVEL'].isel(time=time_step)
        density_at_time = ds_dens.DENS2.isel(time=time_step)
        flow_rate = velocity * cell_thickness * cell_width

        for lat_index in range(ds_vvel.dims['nlat']):
            flow_rate_by_density = np.zeros(len(density_bins) - 1)
            
            for bin_index in range(len(density_bins) - 1):
                in_bin = (density_at_time.isel(nlat=lat_index) >= density_bins[bin_index]) & (density_at_time.isel(nlat=lat_index) < density_bins[bin_index + 1])
                flow_rate_by_density[bin_index] = flow_rate.isel(nlat=lat_index).where(in_bin).sum()
            
            # compute density overturning, reverse to integrate from high to low density
            density_overturning = np.cumsum(flow_rate_by_density)[::-1]
            overturning_dataarray[time_step, lat_index, :] = density_overturning
        
    return overturning_dataarray



da_smoc = calculate_smoc(ds_vvel, ds_dens, density_bins)
ds_smoc = da_smoc.rename({'__xarray_dataarray_variable__': 'sMOC'})

ds_keep = ds_dens.isel(nlon=0).squeeze()
ds_smoc = ds_smoc.assign_coords(TLAT=ds_keep.TLAT)
replacement_value = 0
ds_smoc['TLAT'] = xr.where(
    np.logical_or(np.isnan(ds_smoc['TLAT']), np.isinf(ds_smoc['TLAT']) | np.ma.getmask(ds_smoc['TLAT'])),
    replacement_value,
    ds_smoc['TLAT'])

print('smoc computed')

da_dmoc = (ds_vvel.VVEL * ds_vvel.dz * ds_vvel.DXU).sum(dim='nlon').cumsum(dim='z_t')

ds_dmoc = ds_dmoc.rename({'__xarray_dataarray_variable__': 'dMOC'})
ds_dmoc = ds_dmoc.assign_coords(TLAT=ds_smoc.TLAT)

print('smoc computed')

da_bsf = (ds_vvel.VVEL * ds_vvel.dz * ds_vvel.DXU).sum(dim='z_t').cumsum(dim='nlon')

ds_bsf = ds_bsf.rename({'__xarray_dataarray_variable__': 'BSF'})
ds_bsf = ds_bsf.assign_coords(TLAT=ds_bsf.TLAT)

print('smoc computed')

if period == 'ghg_hist':
    da_smoc.to_netcdf(os.path.join(path, f'Below_combined_smoc_3.0_40_20_ghg_hist.nc'))
    da_dmoc.to_netcdf(os.path.join(path, f'Below_combined_dmoc_3.0_40_20_ghg_hist.nc'))
    da_bsf.to_netcdf(os.path.join(path, f'Below_combined_bsf_3.0_40_20_ghg_hist.nc'))
elif perod == 'above':
    da_smoc.to_netcdf(os.path.join(path, f'Above_combined_smoc_3.0_40_20.nc'))
    da_dmoc.to_netcdf(os.path.join(path, f'Above_combined_dmoc_3.0_40_20.nc'))
    da_bsf.to_netcdf(os.path.join(path, f'Above_combined_bsf_3.0_40_20.nc'))

print('das saved')


