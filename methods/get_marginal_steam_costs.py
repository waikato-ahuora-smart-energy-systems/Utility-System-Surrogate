from pathlib import Path
import pandas as pd

from scripts import Data, Learn, Correlate, SteadyState, Validate, MarginalSteamCosts, SteamCosts, get_plant_allocations, get_steam_costs
from methods.dev_surrogate_model import train_steam_system, train_turbine, train_8PB, train_internal_demands
import numpy as np
import copy
import matplotlib.pyplot as plt 
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder
import pickle
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor  
import concurrent.futures
from multiprocessing import Manager
from joblib import Parallel, delayed
from functools import partial
from joblib.parallel import parallel_backend
import tqdm
from tqdm.contrib.concurrent import process_map, thread_map  # drop-in parallel with progress bar
import os
from itertools import product

# Suppress TensorFlow logs globally
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive


class MSCWorker:
    def __init__(self, msc):
        self.msc = msc

    def __call__(self, args):
        group, reduction = args
        avg_fuel, avg_turbine = self.msc.get_msc(
            group=group,
            reduction=reduction,
            fuel_name='Total NG',
            swing_boiler_name='PB2 Steam Generation',
            turbine_name='Turbine Power Generation'
        )
        return group, reduction, avg_fuel, avg_turbine

def run_marginal_steam_costs():
    # 'MP-3CD' 'LP-IDP'
    plants_to_remove = ['LP-EVP','IP-PM', 'MP-PM', 'MP-PD', 'MP-BPM','MP-CPM', 'LP-CPM', 'MP-BLP', 'LP-BLP', 'LP-CST', ]

    reduction_range = [-0.3]
    
    constants = {
    'dH_steam': 3216, # kJ/kg needed to raise steam from feedwater
    'dH_turbine_MP': 240, # 240  kJ/kg of turbine generation
    'dH_turbine_LP': 460, # 414 kJ/kg of turbine generation
    'fuel_NHV': 42, # MJ/NM3
    'gas_cost': 22.5, # $/GJ gas cost
    'WW_cost': 28.6, # $/wet-tonne WW cost
    'elec_cost': 180, # $/MWh
    'time_interval': 5/60, # time interval in hours
    'plants_to_remove': plants_to_remove # remove LP-AUX and CST since they a supporting plants
    }
    
    
    # Import data. Note that these must be the unfiltered dataset otherwise it will throw mask errors
    gas_data_raw = train_steam_system(process_raw_data=False, filter_data=False, train_new_model=False, validate_model=False) # this will be our basis for filtering the other datsets
    swing_boiler_data_raw = train_8PB(process_raw_data=False, filter_data=False, train_new_model=False, validate_model=False)
    turbine_data_raw = train_turbine(process_raw_data=False, filter_data=False, train_new_model=False, validate_model=False)
    
    # Import filter index - makes sure that the data is aligned across all datasets
    filter_index = pd.read_csv(parent_folder / 'data' / 'filter_index.csv', index_col=0, parse_dates=True)
    # Run MSC procedure
    MSC = MarginalSteamCosts(fuel_data=gas_data_raw, swing_boiler_data=swing_boiler_data_raw, turbine_data=turbine_data_raw, internal_demand_data=None, 
                             constants=constants, common_index=filter_index.index
                             )
    # Load models
    MSC.load_pickles(
        'Total NG', # name of gas model 
        'PB2 Steam Generation', # name of swing boiler model 
        'Turbine Power Generation', # name of turbine model 
    )
    
        # Set up worker
    worker = MSCWorker(MSC)
    
    # Initialize the DataFrame with reductions as columns
    msc_fuel_results_df = pd.DataFrame(index=plants_to_remove, columns=reduction_range)
    msc_turbine_results_df = pd.DataFrame(index=plants_to_remove, columns=reduction_range)
    
    # Run the MSC procedure for each group and reduction
    tasks = list(product(plants_to_remove, reduction_range))
    
    start_time = time.time()

    results = thread_map(worker, tasks, max_workers=8)

    # Post process
    msc_fuel_results_df = pd.DataFrame(index=plants_to_remove, columns=reduction_range)
    msc_turbine_results_df = pd.DataFrame(index=plants_to_remove, columns=reduction_range)

    for group, reduction, avg_fuel, avg_turbine in results:
        msc_fuel_results_df.loc[group, reduction] = avg_fuel
        msc_turbine_results_df.loc[group, reduction] = avg_turbine

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    print(msc_fuel_results_df)
    print(msc_turbine_results_df)
    
    msc_fuel_results_df.to_excel(parent_folder / 'predictions' / 'msc_fuel_results.xlsx', index=True)
    msc_turbine_results_df.to_excel(parent_folder / 'predictions' / 'msc_turbine_results.xlsx', index=True)
    
if __name__ == '__main__': 
    run_marginal_steam_costs()
    
    
    

