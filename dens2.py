#!/usr/bin/env python3
# inagler 28/11/24

import time
import psutil
import os
import glob
import xarray as xr
import gsw
import dask
from dask import delayed

start_time = time.time()

temp_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/TEMP'
salt_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/SALT'
dens_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/DENS2'

# Function to calculate dens2
def calculate_dens2(temp_file, salt_file):
    try: 
        ds_temp = xr.open_dataset(temp_file, chunks={'time': -1})
        ds_salt = xr.open_dataset(salt_file, chunks={'time': -1})

        CT = gsw.conversions.CT_from_pt(ds_salt['SALT'], ds_temp['TEMP'])
        sigma2 = gsw.density.sigma2(ds_salt['SALT'], CT)
        sigma2 = sigma2.rename('SIGMA_2')

        ds_temp.close()
        ds_salt.close()
        return sigma2
    except Exception as e:
        print(f"Error processing: {e}")
        return None

# Function to process a single file
def process_file(temp_file, salt_file, dens_filename):
    try:
        # Calculate dens2
        dens2_data = calculate_dens2(temp_file, salt_file)
        
        if dens2_data is None:
            raise ValueError("Failed to calculate dens2")

        dens2_data = dens2_data.astype('float32')
        dens2_data.attrs['units'] = 'kg/m^3 - 1000'
        dens2_data.attrs['long_name'] = 'Potential Density at 2000 dbar'
        
        # Save the result to the output directory
        output_file = os.path.join(dens_dir, dens_filename)
        dens2_data.to_netcdf(output_file)
        
        print(f'{dens_filename[25:]}  -  complete')
        
    except Exception as e:
        print(f'Error processing {dens_filename[26:]}: {e}')

# Get list of TEMP and SALT files
temp_files = sorted(glob.glob(os.path.join(temp_dir, '*.nc')))
salt_files = sorted(glob.glob(os.path.join(salt_dir, '*.nc')))

# Identify missing DENS2 files
missing_files = []
for temp_file, salt_file in zip(temp_files, salt_files):
    temp_filename = os.path.basename(temp_file)
    dens_filename = temp_filename.replace('TEMP', 'DENS2')
    dens_filepath = os.path.join(dens_dir, dens_filename)
    
    if not os.path.exists(dens_filepath):
        missing_files.append((temp_file, salt_file, dens_filename))

# Create delayed tasks for missing files
tasks = [delayed(process_file)(temp_file, salt_file, dens_filename) for temp_file, salt_file, dens_filename in missing_files]

# Compute the tasks in parallel
dask.compute(*tasks)

print('process complete')    
end_time = time.time()
duration = end_time - start_time
print(f"Execution time: {duration} seconds")        