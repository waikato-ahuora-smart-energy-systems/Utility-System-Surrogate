from pathlib import Path
import pandas as pd

from scripts import Data, Learn, Correlate, SteadyState, Validate, PlantAllocations, get_plant_allocations, get_steam_costs
from methods.dev_surrogate_model import train_natural_gas, train_turbine, train_PB2
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
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from multiprocessing import Manager
from joblib import Parallel, delayed
from functools import partial
from joblib.parallel import parallel_backend
from tqdm.contrib.concurrent import process_map  # drop-in parallel with progress bar
import os

# Suppress TensorFlow logs globally
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive

from itertools import chain, combinations

def generate_valid_subsets(items):
    all_subsets = list(chain.from_iterable(combinations(items, r) for r in range(len(items) + 1)))
    valid_subsets = []
    
    for subset in all_subsets:
        subset = set(subset)
        
        ''' '''
        # Remove single item and null subsets
        if len(subset) == 1 or len(subset) == 0:
            continue
        
        '''  '''
        # Constraint 1: BLP_sum can only be in a subset with PD_sum or CPM_sum. Doesnt apply in reverse if theres UKP runs
        if "BLP_sum" in subset and ("PD_sum" not in subset or "CPM_sum" not in subset):
            continue
        
        # Constraint 2: The exact subset {CEP_sum, BLP_sum, HPM_sum} and its direct pairings cannot exist since they are too small to operate independently
        forbidden_combinations = [{"CEP_sum", "BLP_sum", "HPM_sum"},
                                  {"CEP_sum", "BLP_sum"},
                                  {"CEP_sum", "HPM_sum"},
                                  {"BLP_sum", "HPM_sum"}]
        
        if subset in forbidden_combinations:
            continue
        
        ''' '''
        # Constraint 3: CEP_sum can only be in a subset with PD_sum or CPM_sum
        if "CEP_sum" in subset and ("PD_sum" not in subset or "CPM_sum" not in subset):
            continue
        
        valid_subsets.append(subset)
    
    return valid_subsets

def setup_data():
    
    # Import data. Note that these must be the unfiltered dataset otherwise it will throw mask errors
    gas_data = train_natural_gas(process_raw_data=False, filter_data=False, train_new_model=False, validate_model=False) # this will be our basis for filtering the other datsets
    swing_boiler_data = train_PB2(process_raw_data=False, filter_data=False, train_new_model=False, validate_model=False)
    turbine_data = train_turbine(process_raw_data=False, filter_data=False, train_new_model=False, validate_model=False)

    # Retain rows that had all plants running at the same time
    gas_data = gas_data[
                (gas_data['CPM On/Off'] > 0) & 
                (gas_data['PD On/Off'] > 0) &
                (gas_data['BPM On/Off'] > 0) &
                (gas_data['PM On/Off'] > 0) &
                (gas_data['HPM On/Off'] > 0) 
                ]

    # Remove data outside of floor & ceiling
    gas_data = gas_data[
    (gas_data['PM_sum'].between(45, 135)) &
    (gas_data['PD_sum'].between(20, 45)) &
    (gas_data['BPM_sum'].between(10, 100)) &
    (gas_data['CPM_sum'].between(15, 75)) &
    (gas_data['BLP_sum'].between(0, 25))
    ]
   
    gas_data.drop(columns=['Total NG'], inplace=True)
    swing_boiler_data.drop(columns=['PB2 Steam Generation'], inplace=True) 
    turbine_data.drop(columns=['Turbine Power Generation'], inplace=True)
    
    # Remove rows in the other data that were filtered out in the gas data
    common_time_index = gas_data.index.intersection(swing_boiler_data.index)
    swing_boiler_data = swing_boiler_data.loc[common_time_index]
    turbine_data = turbine_data.loc[common_time_index]
    internal_demand_data = None
    #internal_demand_data = internal_demand_data.loc[common_time_index]
    gas_data.to_excel(parent_folder / 'predictions' / 'Gas Data.xlsx', index=False) # save the gas data to excel for testing purposes
    return gas_data, swing_boiler_data, turbine_data, internal_demand_data

def modify_plant_demand(plant_name, gas_data, swing_boiler_data, turbine_data, internal_demand_data, reduction_pct):
    # Reduce the plant demand by a certain percentage

    # Reduce data in the gas demand datasets
    gas_data["Total Steam Demand"] = gas_data["Total Steam Demand"] + (gas_data[plant_name] * reduction_pct)
    gas_data[plant_name] = gas_data[plant_name] * (1 + reduction_pct)
    
    # Reduce data in swing boiler dataset
    swing_boiler_data["Total Steam Demand"] = swing_boiler_data["Total Steam Demand"] + (swing_boiler_data[plant_name] * reduction_pct)
    swing_boiler_data[plant_name] = swing_boiler_data[plant_name] * (1 + reduction_pct)
    
    if plant_name == 'BPM_sum': # modify the lagged data for BPM_sum
        swing_boiler_data["BPM_sum--1"] = swing_boiler_data["BPM_sum"].shift(1) 
    
    # Reduce data in turbine dataset ## look at scenario modelliong version of turbine model and remove MP, LP features
    turbine_data["Total Steam Demand"] = turbine_data["Total Steam Demand"] + (turbine_data[plant_name] * reduction_pct)
    turbine_data["Total Steam Demand--1"] = turbine_data["Total Steam Demand"].shift(1)
    turbine_data[plant_name] = turbine_data[plant_name] * (1 + reduction_pct)
    
    return gas_data, swing_boiler_data, turbine_data, internal_demand_data   

def run_plant_allocations(gas_data, swing_boiler_data, turbine_data, internal_demand_data, valid_subsets, constants, scenario, mode):
    # Calculate the system flows
    steam_costs = PlantAllocations(gas_data, swing_boiler_data, turbine_data, internal_demand_data, valid_subsets, constants, scenario)
    
    steam_costs.load_pickles(
        'Total NG', # name of gas model 
        'PB2 Steam Generation', # name of swing boiler model 
        'Turbine Power Generation', # name of turbine model 
    )  
    
    steam_costs.get_system_flows(mode=mode)
    
    # Calculate the cost allocations
    start_time = time.time()
   
    # Convert Pandas rows to serializable format (dicts) and add row index for tracking
    args_list = [
        (
            row_time, row_system_flows.to_dict(),  
            steam_costs.gas_cost, steam_costs.elec_cost, steam_costs.dH_steam, steam_costs.dH_turbine_LP ,steam_costs.time_interval, 
            steam_costs.plants_to_remove, steam_costs.subset_combinations, 
            gas_data.loc[row_time].to_dict(),  
            mode,  
            row_number  
        )
        for row_number, (row_time, row_system_flows) in enumerate(steam_costs.system_flows.iterrows(), start=1)
    ]

    # Parallel execution using joblib
    fuel_results = Parallel(n_jobs=8, backend="loky", prefer="processes")(
        delayed(get_plant_allocations)(*args) for args in args_list)

    end_time = time.time()
    
    print(f"execution time: {end_time - start_time:.2f} seconds")
    steam_costs.postprocess_results(fuel_results, mode=mode)  

if __name__ == "__main__":
    # Constants
    plants_to_remove = ['PM_sum', 'PD_sum', 'BPM_sum', 'CPM_sum', 'BLP_sum', 'HPM_sum', 'CEP_sum']
    constants = {
    'dH_steam': 3216, # kJ/kg needed to raise steam from feedwater
    'dH_turbine_MP': 240, # 240  kJ/kg of turbine generation
    'dH_turbine_LP': 460, # 414 kJ/kg of turbine generation
    'fuel_NHV': 42, # MJ/NM3
    'gas_cost': np.nan , # $/GJ gas cost
    'WW_cost': np.nan, # $/wet-tonne WW cost
    'elec_cost': np.nan, # $/MWh
    'time_interval': 5/60, # time interval in hours
    'plants_to_remove': plants_to_remove # remove S&R and C&K since they a supporting plants
    }
    

    mode = 'Costs' # 'Revenues' #'Costs' # switch to 'Revenues' to get the revenues or 'Costs' to get the costs
    get_base_flows = False # True to get the base flows from previously saved csv, False to calculate them fresh
    
    # Get the data
    gas_data, swing_boiler_data, turbine_data, internal_demand_data = setup_data()
    
    # Calculate the base cost allocations  
    if get_base_flows:   
        # Get plant subsets
        valid_subsets = generate_valid_subsets(plants_to_remove)
        run_plant_allocations(gas_data, swing_boiler_data, turbine_data, internal_demand_data, valid_subsets, constants, scenario="Base", mode=mode)
    
