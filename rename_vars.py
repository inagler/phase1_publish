#!/usr/bin/env python3
# inagler 15/01/25
# rename_vars.py

import xarray as xr
import os
import glob

# Directory where the DENS2 files are located
dens_dir = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/DENS2'

# Function to rename variable in DataArray
def rename_variable(filename):
    try:
        # Open the dataset as a DataArray (if it contains only a DataArray)
        data_array = xr.open_dataarray(filename)
        
        if 'SIGMA_2' in data_array.name:
            # Rename variable
            data_array = data_array.rename('DENS2')
            # Save to the same file or a new file
            data_array.to_netcdf(filename)  # Overwrite the existing one
            print(f'Renamed SIGMA_2 to DENS2 in {filename}')
        else:
            print(f'SKIP: {filename} does not contain SIGMA_2')
        
    except Exception as e:
        print(f'Error processing {filename}: {e}')

# Get list of DENS2 files
dens_files = sorted(glob.glob(os.path.join(dens_dir, '*.nc')))

# Rename variable for each file
for dens_file in dens_files:
    rename_variable(dens_file)

print("Variable renaming complete")