'''
__author__ = 'Keegan Hall'
__credits__ = 

Class for using trained machine learning of industrial plant data to predict marginal steam costs
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
import copy
#from gekko import GEKKO
import itertools
import numpy as np
import re

parent_folder = Path(__file__).parent.parent # location of the folder containing this file relative to your C drive

class MarginalSteamCosts():
    def __init__(
        self,
        fuel_data, # fuel data
        swing_boiler_data, # swing boiler data
        turbine_data, # turbine data
        internal_demand_data, # internal demand data
        constants,  # constans dictionary
        common_index, # common index for all data
        ): 
        '''Class init'''
        
        # Unpack data & make copies
        self.fuel_data = fuel_data
        self.swing_boiler_data = swing_boiler_data
        self.turbine_data = turbine_data
        self.internal_demand_data = internal_demand_data
        self.common_index = common_index
      
        self.fuel_data_backup = copy.deepcopy(fuel_data)
        self.swing_boiler_data_backup = copy.deepcopy(swing_boiler_data)
        self.turbine_data_backup = copy.deepcopy(turbine_data)
        self.internal_demand_data_backup = copy.deepcopy(internal_demand_data)
        
        # Unpack constants
        self.dH_steam = constants['dH_steam'] # kJ/kg needed to raise steam from feedwater
        self.dH_turbine_MP = constants['dH_turbine_MP'] # kJ/kg of turbine generation
        self.dH_turbine_LP = constants['dH_turbine_LP'] # kJ/kg of turbine generation
        self.fuel_NHV = constants['fuel_NHV'] # MJ/NM3
        self.gas_cost = constants['gas_cost'] # $/GJ gas cost
        self.WW_cost = constants['WW_cost'] # $/wet-tonne WW cost
        self.elec_cost = constants['elec_cost'] # $/MWh
        self.time_interval = constants['time_interval'] # time interval in hours
        self.plants_to_remove = constants['plants_to_remove']
    
    def load_pickles(
        self,          
        gas_name, # name of gas model 
        swing_boiler_name, # name of swing boiler model 
        turbine_name, # name of turbine model 
        internal_demand_name, # name of internal demand model
        ): 
        '''
        Open pickle ML model

        Open previously trained model from pickle 
        '''
        # Open pickles
        self.fuel_model = pickle.load(open(parent_folder / 'models' / '{} Model.pkl'.format(gas_name) ,'rb'))
        self.swing_boiler_model = pickle.load(open(parent_folder /  'models' / '{} Model.pkl'.format(swing_boiler_name) ,'rb'))
        self.turbine_model = pickle.load(open(parent_folder / 'models' / '{} Model.pkl'.format(turbine_name)  ,'rb'))
        
    def get_msc(self, group, reduction, fuel_name, swing_boiler_name, turbine_name):
        '''
        Run the MSC procedure 

        Use pre-trained ML models to predict the fuel cost for all time points
        '''
        
        msc_fuel, average_msc_fuel = self.calculate_fuel_costs(group=group, reduction=reduction, fuel_name=fuel_name, swing_boiler_name=swing_boiler_name)
        msc_turbine, average_msc_turbine = self.calculate_turbine_revanue(group=group, reduction=reduction, turbine_name=turbine_name)
        
        return average_msc_fuel, average_msc_turbine
        
    def predict_fuel_consumption(
        self,
        data, 
        group, # header name of steam group to change
        reduction, # fraction change to apply to steam group data
        target_col_name, # name of target column in data
        swing_boiler_gen): 
        '''
        Calculates fuel cost 

        Use pre-trained ML models to predict the fuel cost for all time points
        '''
        # Remove target column
        data = data.drop(columns=[target_col_name])
        
        # Reduce steam load
        steam_change = data[group] * reduction
        data[group] += steam_change
        data["Total Steam Demand"] += steam_change
        
        # Shift steam back
        data["Total Steam Demand--1"] = data["Total Steam Demand"].shift(1)    
        
        # Filter data
        data = data.loc[data.index.isin(self.common_index)] 
        #data = data[data[group] + steam_change > 5]
        
        # Add swing boiler generation
        data['PB2 Steam Generation'] = swing_boiler_gen # set swing boiler generation to predicted swing boiler generation

        # Predict fuel consumption in $/interval
        fuel_consumption = np.array(self.fuel_model.predict(data)) # total gas consumption in NM3/hr
        fuel_cost = fuel_consumption*self.fuel_NHV/1000 * self.gas_cost * self.time_interval  
        
        return fuel_cost

    def predict_swing_boiler_gen(
        self,
        data, 
        group, # header name of steam group to change
        reduction, # fraction change to apply to steam group data
        target_col_name, # name of target column in data
        ): 
        '''
        Calculates swing boiler generation for a subset

        Use pre-trained ML models to predict the swing boiler generation of a particular subset of plants for all time points
        '''
        
        # Remove target column
        data = data.drop(columns=[target_col_name])
        
        # Reduce steam load
        steam_change = data[group] * reduction
        data[group] += steam_change
        data["Total Steam Demand"] += steam_change
        
        # Shift steam back
        data['MP-BPM'] = data['MP-BPM'].shift(1)
        data["Total Steam Demand--1"] = data["Total Steam Demand"].shift(1)    
        
        # Filter data
        data = data.loc[data.index.isin(self.common_index)] 

        # Predict fuel consumption in $/interval
        swing_boiler_gen = np.array(self.swing_boiler_model.predict(data)) # total gas consumption in NM3/hr
       
        return swing_boiler_gen

    def predict_power_generation(
        self,
        data, 
        group, # header name of steam group to change
        reduction, # fraction change to apply to steam group data
        target_col_name, # name of target column in data
        ):
        
        # Remove target column
        data = data.drop(columns=[target_col_name])
        
        # Reduce steam load
        steam_change = data[group] * reduction
        data[group] += steam_change
        data["Summed Steam Demand"] += steam_change
        
        # Shift steam back
        data["Summed Steam Demand--1"] = data["Summed Steam Demand"].shift(1)    
        
        # Filter data
        data = data.loc[data.index.isin(self.common_index)] 
    
        # Predict fuel consumption in $/interval
        turbine_generation = np.array(self.turbine_model.predict(data)) # total power generation in MW
        power_revanue = turbine_generation* self.elec_cost * self.time_interval  
        
        return power_revanue

    def calculate_fuel_costs(
        self,
        group, # header name of steam group to change
        reduction, # fraction change to apply to steam group data
        fuel_name,
        swing_boiler_name):
        '''
        Predict change in operating cost

        Use previously trained model to predict fuel use in response to a change in steam group demand
        '''
        # Predict swing boiler generation
        baseline_swing_boiler_gen = self.predict_swing_boiler_gen(data=self.swing_boiler_data.copy(), group=group, reduction=0, target_col_name=swing_boiler_name) # predict swing boiler generation
        reduced_load_swing_boiler_gen = self.predict_swing_boiler_gen(data=self.swing_boiler_data.copy(), group=group, reduction=reduction, target_col_name=swing_boiler_name) # predict swing boiler generation
        
        # Predict fuel costs
        baseline_fuel_costs = self.predict_fuel_consumption(data=self.fuel_data.copy(), group=group, reduction=0, target_col_name=fuel_name, swing_boiler_gen=baseline_swing_boiler_gen) 
        reduced_load_fuel_costs = self.predict_fuel_consumption(data=self.fuel_data.copy(), group=group, reduction=reduction, target_col_name=fuel_name, swing_boiler_gen=reduced_load_swing_boiler_gen)
        
        baseline_costs = pd.Series(baseline_fuel_costs, index=self.common_index)
        reduced_costs = pd.Series(reduced_load_fuel_costs, index=self.common_index)
        
        # Calculate marginal steam fuel cost ($/t)
        steam_col = self.fuel_data.loc[self.fuel_data.index.isin(self.common_index), group]
        denominator = steam_col * abs(reduction) * self.time_interval
        denominator = denominator.replace(0, pd.NA)
        threshold = self.fuel_data[group].nlargest(20).iloc[-1] * 0.12 # filter out steam loads that are less than 10% of the max steam load
        if group == 'MP-BPM': # this is a special case where there the sensor had an error and can't be removed via automated filtering
            threshold = 110*0.12 
        mask = steam_col > threshold
        denominator_filtered = denominator[mask]
        
        msc_fuel = (baseline_costs[mask] - reduced_costs[mask]).divide(denominator_filtered)
        average_msc_fuel = msc_fuel.mean()
        
        return msc_fuel, average_msc_fuel
    
    def calculate_turbine_revanue(
        self,
        group, # header name of steam group to change
        reduction, # fraction change to apply to steam group data
        turbine_name):
        '''
        Predict change in operating cost

        Use previously trained model to predict fuel use in response to a change in steam group demand
        '''
        
        # Predict turbine revanue 
        baseline_turbine_revanue = self.predict_power_generation(data=self.turbine_data.copy(), group=group, reduction=0, target_col_name=turbine_name) 
        reduced_turbine_revanue = self.predict_power_generation(data=self.turbine_data.copy(), group=group, reduction=reduction, target_col_name=turbine_name)
        
        # Calculate marginal steam turbine revanue ($/t)
        baseline_revenue = pd.Series(baseline_turbine_revanue, index=self.common_index)
        reduced_revenue = pd.Series(reduced_turbine_revanue, index=self.common_index)
        
        # Calculate marginal steam fuel cost ($/t)
        steam_col = self.fuel_data.loc[self.fuel_data.index.isin(self.common_index), group]
        denominator = steam_col * abs(reduction) * self.time_interval
        denominator = denominator.replace(0, pd.NA)
        threshold = self.fuel_data[group].nlargest(20).iloc[-1] * 0.12 # filter out steam loads that are less than 10% of the max steam load
        if group == 'MP-BPM': # this is a special case where there the sensor had an error and can't be removed via automated filterin
            threshold = 110*0.12 
        mask = steam_col > threshold
        denominator_filtered = denominator[mask]
        msc_turbine = (baseline_revenue[mask] - reduced_revenue[mask]).divide(denominator_filtered)
        average_msc_turbine = msc_turbine.mean()
        
        return msc_turbine, average_msc_turbine


        