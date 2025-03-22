#!/usr/bin/env python3
# inagler 25/11/24

import time
import os
import sys
import glob
import csv
import xarray as xr



def main():

    start_time = time.time()
    
    
    if len(sys.argv) < 6:  # Expect 5 arguments beyond the script name
        print("Usage: python change_point_analysis.py <P1_len> <P2_len> <threshold_multiple> <gap> <detrended>")
        sys.exit(1)

    # Parse the arguments
    P1_len = int(sys.argv[1])
    P2_len = int(sys.argv[2])
    threshold_multiple = float(sys.argv[3])
    gap = int(sys.argv[4])
    event_period = P1_len + P2_len

    # Parse the detrended flag
    detrended = sys.argv[5].lower() == 'true'

    # load files
    smoc_dir = '/Data/skd/scratch/innag3580/comp/smoc/'
    smoc55_files = sorted(glob.glob(os.path.join(smoc_dir, 'smoc55_member_*.nc')))
    # define historical period
    #hist_end = (2014-1850)*12
    hist_end = (2030-1850)*12

    print(f"""
    Change point analysis started
    P1 length = {P1_len}
    P2 length = {P2_len}
    Threshold multiple = {threshold_multiple}
    Gap = {gap}
    """)

    if detrended == True:
        print('DETRENDED')
        print('')
        file_name = 'smoc55_ensemble_mean.nc'
        smoc55_mean_file = os.path.join(smoc_dir, file_name)


    ### COMPUTATION ### 

    criteria_indices_dict = {}
    for i in range(len(smoc55_files)):

        # load data and compute annual mean
        da = xr.load_dataarray(smoc55_files[i]).isel(time=slice(0, hist_end))
        annual_mean = da.resample(time="1Y").mean()
        if detrended == True:
            ensemble_mean_da = xr.load_dataarray(smoc55_mean_file).isel(time=slice(0, hist_end))
            annual_ensemble_mean = ensemble_mean_da.resample(time="1Y").mean()
            annual_mean = annual_mean - annual_ensemble_mean

        # prepare storage
        criteria_indices = []

        # Start after first window size
        for j in range(P1_len, len(annual_mean) - P2_len):

            # Compute mean and std of comparison window
            P1 = annual_mean[j - P1_len:j]
            P1_std = P1.std().item()
            P1_avg = P1.mean().item()

            # Compute mean of range of interest
            P2 = annual_mean[j:j + P2_len]
            P2_avg = P2.mean().item()

            # Check if the std is above the threshold
            if P2_avg >= (P1_avg + (threshold_multiple * P1_std)):
                condition = "Above"
            elif P2_avg <= (P1_avg - (threshold_multiple * P1_std)):
                condition = "Below"
            else:
                condition = "Within"

            if condition != "Within":
                if len(criteria_indices) == 0:
                    criteria_indices.append((j, condition))
                elif j >= criteria_indices[-1][0] + gap:
                    criteria_indices.append((j, condition))

        # Save criteria_indices in the dictionary
        criteria_indices_dict[smoc55_files[i][-11:-3]] = criteria_indices

    result = []
    for member, values in criteria_indices_dict.items():
        if values:
            result.append((member, values))

    ### OUTPUT ###

    # Define the output directory and filename
    output_dir = os.path.join(os.environ['HOME'], 'phase1_CONDA/publishable_code')
    os.makedirs(output_dir, exist_ok=True)
    if detrended == True:
        output_filename = f"change_point_indices_{threshold_multiple}_{P1_len}_{P2_len}_{gap}-detrended.csv"
    else:
        output_filename = f"plotting-change_point_indices_{threshold_multiple}_{P1_len}_{P2_len}_{gap}.csv"
        #output_filename = f"change_point_indices_{threshold_multiple}_{P1_len}_{P2_len}_{gap}.csv"

    output_filepath = os.path.join(output_dir, output_filename)


    with open(output_filepath, 'w', newline='') as csvfile:
        fieldnames = ['Member', 'Values', 'Condition']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for member, values in result:
            for value, condition in values:
                writer.writerow({'Member': member, 'Values': value, 'Condition': condition})

    print(f"""Indices saved to {output_filepath}""")
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration} seconds")
    
    
if __name__ == "__main__":
    main()