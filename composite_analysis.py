#!/usr/bin/env python3
# inagler 28/11/24

import time  # For measuring execution time
import os  # For interacting with the operating system (e.g., file paths)
import glob  # For file pattern matching

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import xarray as xr  # For working with labeled multi-dimensional arrays

import sys
import dask
from dask import delayed, compute  # For parallel computing with delayed execution

import cftime  # For handling time in netCDF files
import pop_tools  # For working with the POP data

import warnings
from dask.array import PerformanceWarning

# Suppress the specific performance warning globally
warnings.filterwarnings("ignore", category=PerformanceWarning)

def standardise_time(ds):
    ds['time'] = xr.decode_cf(ds, use_cftime=True).time
    if isinstance(ds.time.values[0], cftime._cftime.DatetimeNoLeap):
        time_as_datetime64 = np.array([pd.Timestamp(str(dt)).to_datetime64() for dt in ds.time.values])
        ds['time'] = xr.DataArray(time_as_datetime64, dims='time')
    return ds

def DJFM_average(ds):
    numeric_vars = {k: v for k, v in ds.data_vars.items() if np.issubdtype(v.dtype, np.number)}
    ds_numeric = xr.Dataset(numeric_vars, coords=ds.coords)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_first_FM = ds_numeric.isel(time=slice(0, 2)).coarsen(time=2, boundary='trim').mean()
        ds_DJFM = ds_numeric.isel(time=slice(2, None)).coarsen(time=4, boundary='trim').mean()
    ds_combined = xr.concat([ds_first_FM, ds_DJFM], dim='time')
    return ds_combined

def prepare_ds_member(var, member_id, period):
    if period == 'hist':
        file_pattern = os.path.join(base_path, var, f'*BHIST*LE2-{member_id}*.nc')
        file_paths = sorted(glob.glob(file_pattern))
    elif period == 'aa_hist':
        file_patterns = [
            os.path.join(base_path, var, f'*{member_id}*.{var}.185001-185912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.186001-186912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.187001-187912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.188001-188912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.189001-189912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.190001-190912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.191001-191912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.192001-192912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.193001-193912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.194001-194912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.195001-195912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.196001-196912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.197001-197912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.198001-198912.nc'),
        ]
    elif period == 'ghg_hist':
        file_patterns = [
            os.path.join(base_path, var, f'*{member_id}*.{var}.193001-193912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.194001-194912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.195001-195912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.196001-196912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.197001-197912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.198001-198912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.199001-199912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.200001-200912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.201001-201412.nc')
        ]
    elif period == 'early_hist':
        
        print('yay early_hist')
        
        file_patterns = [
            os.path.join(base_path, var, f'*{member_id}*.{var}.185001-185912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.186001-186912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.187001-187912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.188001-188912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.189001-189912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.190001-190912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.191001-191912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.192001-192912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.193001-193912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.194001-194912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.195001-195912.nc'),
        ]
    elif period == 'late_hist':
        file_patterns = [
            os.path.join(base_path, var, f'*{member_id}*.{var}.190001-190912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.191001-191912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.192001-192912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.193001-193912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.194001-194912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.195001-195912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.196001-196912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.197001-197912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.198001-198912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.199001-199912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.200001-200912.nc'),
            os.path.join(base_path, var, f'*{member_id}*.{var}.201001-201412.nc')
        ]
        
    if period in ['aa_hist', 'ghg_hist', 'early_hist', 'late_hist']:
        filtered_files = []
        for pattern in file_patterns:
            filtered_files.extend(glob.glob(pattern))
        file_paths = sorted(filtered_files)

    ds_member = xr.open_mfdataset(file_paths, chunks={'time': 120}, preprocess=standardise_time)
    ds_member[var] = ds_member[var].astype('float32')
    ds_member = ds_member.sel(time=ds_member['time.month'].isin([12, 1, 2, 3]))
    ds_member = DJFM_average(ds_member)
    if var != 'N_HEAT':
        ds_member = ds_member.roll(nlon=-100).where(NA_mask == 1)
    return ds_member   

def extract_composite(ds_member, time_slice, index):
    ds = ds_member.isel(time=time_slice)
    new_time = xr.DataArray(range(len(ds.time)), dims='time')
    ds = ds.assign_coords(time=new_time)
    save_name = f'{threshold_multiple}_{P1_len}_{P2_len}_{index}.nc'
    ds.to_netcdf(os.path.join(temporary_path, save_name))
    ds.close()
    
def process_member(member, group_data, var, period):
    member_id = f"{float(member):.3f}"
    ds_member = prepare_ds_member(var, member_id, period)
    for index, event in zip(group_data.index, group_data['Values']):
        event_time = event
        period_start = event_time - P1_len
        period_end = event_time + P2_len
        if period == 'ghg_hist':
            time_slice = slice(period_start-79, period_end-79)
        else:     
            time_slice = slice(period_start, period_end)
        extract_composite(ds_member, time_slice, index)
    print('member evaluated')

def combine_and_cleanup(split_idx, var):
    file_paths = glob.glob(os.path.join(temporary_path, f'{threshold_multiple}_{P1_len}_{P2_len}_*.nc'))
    above_files = [fp for fp in file_paths if df.loc[int(fp.split('_')[-1].split('.')[0]), 'Condition'] == 'Above']
    below_files = [fp for fp in file_paths if df.loc[int(fp.split('_')[-1].split('.')[0]), 'Condition'] == 'Below']
    
    if above_files:
        save_name = f'Above_{split_idx}_{var}_{threshold_multiple}_{P1_len}_{P2_len}.nc'
        combined_above = xr.open_mfdataset(above_files, concat_dim='new_dim', combine='nested').mean(dim='new_dim')
        combined_above.to_netcdf(os.path.join(temporary_path, save_name))
        for file_path in above_files:
            os.remove(file_path)
            print('deleted above files')

    if below_files:
        save_name = f'Below_{split_idx}_{var}_{threshold_multiple}_{P1_len}_{P2_len}.nc'
        combined_below = xr.open_mfdataset(below_files, concat_dim='new_dim', combine='nested').mean(dim='new_dim')
        combined_below.to_netcdf(os.path.join(temporary_path, save_name))
        for file_path in below_files:
            os.remove(file_path)
            print('deleted below files')

def main():
    start_time = time.time()

    if len(sys.argv) < 4:
        print("Usage: python your_script.py <var> <filename> <period>")
        sys.exit(1)

    var = sys.argv[1]
    filename = sys.argv[2]
    period = sys.argv[3]
    
    # Declare global variables that will be modified
    global threshold_multiple, P1_len, P2_len, NA_mask, base_path, temporary_path, final_path, df

    # Load data from change point analysis
    path = os.path.join(os.environ['HOME'], 'phase1_CONDA/publishable_code')
    file = os.path.join(path, filename)
    df = pd.read_csv(file)
    
    if period == 'hist':
        df = df
    elif period == 'aa_hist':
        df = df[df['Condition'] == 'Below']
        df = df[df['Values'] < 119]
    elif period == 'ghg_hist':
        df = df[df['Condition'] == 'Below']
        df = df[df['Values'] >= 119]
    elif period == 'early_hist':
        df = df[df['Condition'] == 'Above']
        df = df[df['Values'] < 89]
    elif period == 'late_hist':
        df = df[df['Condition'] == 'Above']
        df = df[df['Values'] >= 89]

    # Extract the variables from the filename
    filename = os.path.basename(file)
    parts = filename.replace('change_point_indices_', '').replace('.csv', '').split('_')
    threshold_multiple = float(parts[0])
    P1_len = int(parts[1])
    P2_len = int(parts[2])

    # set up mask
    grid_name = 'POP_gx1v7'
    region_defs = {
        'subzero_Atlantic':[
            {'match': {'REGION_MASK': [6]}, 'bounds': {'TLAT': [10.0, 80.0], 'TLONG': [260.0, 360.0]}}
        ],
        'superzero_Atlantic':[
            {'match': {'REGION_MASK': [6]}, 'bounds': {'TLAT': [10.0, 80.0], 'TLONG': [0, 20.0]}}
        ],
        'Mediterranean': [
            {'match': {'REGION_MASK': [7]}}
        ],
        'LabradorSea': [
            {'match': {'REGION_MASK': [8]}, 'bounds': {'TLAT': [10.0, 80.0]}}
        ],
            'NordicSea': [
            {'match': {'REGION_MASK': [9]}, 'bounds': {'TLAT': [10.0, 80.0]}}
        ]
    }
    NA_mask = pop_tools.region_mask_3d(grid_name, region_defs=region_defs, mask_name='North Atlantic Mask')
    NA_mask = NA_mask.sum('region')
    NA_mask = NA_mask.roll(nlon=-100)

    # set up paths
    base_path = '/Data/gfi/share/ModData/CESM2_LENS2/ocean/monthly/'
    temporary_path = '/Data/skd/scratch/innag3580/comp/temporary/'
    final_path = '/Data/skd/scratch/innag3580/comp/composites/'

    ### OUTPUT ###
    print(f"""
    Composite computation started
    variable = {var}
    P1 length = {P1_len}
    P2 length = {P2_len}
    Threshold multiple = {threshold_multiple}
    period = {period}
    """)

    # Split the DataFrame
    if len(df) > 100:
        split_number = 3
    else:
        split_number = 1
    split_size = len(df) // split_number
    splits = [df[i:i + split_size] for i in range(0, len(df), split_size)]

    # Process each split with an index
    for split_idx, split in enumerate(splits):
        
        print(split_idx, ' \ ', split_number)
        
        grouped = split.groupby('Member')
        #tasks = [dask.delayed(process_member)(member, group_data, var, period) for member, group_data in grouped]
        tasks = [dask.delayed(process_member)(member, group_data, var, period) for member, group_data in grouped
                if not (var == 'VVEL' and (member == 1231.002 or member == 1281.012))]
        dask.compute(*tasks)
        combine_and_cleanup(split_idx, var)
 
    # Combine all 'Above' files
    above_files = glob.glob(os.path.join(temporary_path, f'Above_*_{var}_{threshold_multiple}_{P1_len}_{P2_len}.nc'))
    if above_files:
        combined_above = xr.open_mfdataset(above_files, concat_dim='new_dim', combine='nested').mean(dim='new_dim')
        combined_above.to_netcdf(os.path.join(final_path, f'Above_combined_{var}_{threshold_multiple}_{P1_len}_{P2_len}_{period}.nc'))
        for file_path in above_files:
            os.remove(file_path)
        print('deleted final above files')
    else:
        print('no above files')
    
    # Combine all 'Below' files
    below_files = glob.glob(os.path.join(temporary_path, f'Below_*_{var}_{threshold_multiple}_{P1_len}_{P2_len}.nc'))
    if below_files:
        combined_below = xr.open_mfdataset(below_files, concat_dim='new_dim', combine='nested').mean(dim='new_dim')
        combined_below.to_netcdf(os.path.join(final_path,f'Below_combined_{var}_{threshold_multiple}_{P1_len}_{P2_len}_{period}.nc'))
        for file_path in below_files:
            os.remove(file_path)
        print('deleted below files')
    else:
        print('no below files')
    
    print('process complete')     
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration} seconds")        

if __name__ == "__main__":
    main()
    
    