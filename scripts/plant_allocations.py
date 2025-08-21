'''
__author__ = 'Keegan Hall'
__credits__ = 

Class for using trained machine learning of industrial plant data to assign costs to different plants
using game theory
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
from gekko import GEKKO
import itertools
import numpy as np

parent_folder = Path(__file__).parent.parent # location of the folder containing this file relative to your C drive

class PlantAllocations():
    def __init__(
        self,
        fuel_data, # fuel data
        swing_boiler_data, # swing boiler data
        turbine_data, # turbine data
        internal_demand_data, # internal demand data
        subset_combinations, # all feasible subsets of plants
        constants,  # constans dictionary
        scenario, # scenario name
        ): 
        '''Class init'''
        
        # Unpack data & make copies
        self.fuel_data = fuel_data
        self.swing_boiler_data = swing_boiler_data
        self.turbine_data = turbine_data
        self.internal_demand_data = internal_demand_data
      
        
        self.fuel_data_backup = copy.deepcopy(fuel_data)
        self.swing_boiler_data_backup = copy.deepcopy(swing_boiler_data)
        self.turbine_data_backup = copy.deepcopy(turbine_data)
        self.internal_demand_data_backup = copy.deepcopy(internal_demand_data)
        
        # Unpack constants
        self.subset_combinations = subset_combinations
        self.dH_steam = constants['dH_steam'] # kJ/kg needed to raise steam from feedwater
        self.dH_turbine_MP = constants['dH_turbine_MP'] # kJ/kg of turbine generation
        self.dH_turbine_LP = constants['dH_turbine_LP'] # kJ/kg of turbine generation
        self.fuel_NHV = constants['fuel_NHV'] # MJ/NM3
        self.gas_cost = constants['gas_cost'] # $/GJ gas cost
        self.WW_cost = constants['WW_cost'] # $/wet-tonne WW cost
        self.elec_cost = constants['elec_cost'] # $/MWh
        self.time_interval = constants['time_interval'] # time interval in hours
        self.plants_to_remove = constants['plants_to_remove']
        self.scenario = scenario
        
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
    
    def get_system_flows(
        self, 
        mode,
        ):
        '''
        Calculates system flows for all subsets for all time points

        Calculate the system costs by iterating through each possible subset of plants including the total system to 
        '''
        # Check for pre-existing system flows
        file_path = parent_folder / "predictions" / f"System {mode} {self.scenario}.csv"
        
        if file_path.exists(): # retrieve from file
            self.system_flows = pd.read_csv(file_path, index_col=0)
            print("System flows already exist, loading from file...")
            
        else: # calculate system flows
            # Setup dataframes
            self.system_flows = pd.DataFrame(index=self.fuel_data.index, columns=['total system flow']+[', '.join(subset) for subset in self.subset_combinations if len(subset) < len(self.plants_to_remove)]) # system costs for each possible subset of plants

            # Calculate system cost for each possible subset of plants
            for subset in self.subset_combinations:
                # Create fresh copy of datasets for each subsets
                fuel_data_copy = copy.deepcopy(self.fuel_data) # use fresh copy of data for each subset
                swing_boiler_data_copy = copy.deepcopy(self.swing_boiler_data)
                turbine_data_copy = copy.deepcopy(self.turbine_data)
                internal_demand_data_copy = copy.deepcopy(self.internal_demand_data)
                
                # Predict internal demand and swing boiler generation
                #internal_demand = self.predict_internal_demand(data=internal_demand_data_copy, subset=subset) # internal demand data
                predicted_swing_boiler_gen = self.predict_swing_boiler_gen(data=swing_boiler_data_copy, subset=subset, internal_demand=None) # swing boiler data
                
                # Make cost/revanue predictions
                if mode == 'Costs':
                    # Predict the system cost/revenue for the subset & add to resepctive column
                    subset_system_flow = self.predict_system_costs(data=fuel_data_copy, subset=subset, internal_demand=None, swing_boiler_gen=predicted_swing_boiler_gen) 
                elif mode == 'Revenues':
                    subset_system_flow = self.predict_system_revanue(data=turbine_data_copy, subset=subset)
                
                # Save to dataframe
                if len(subset) == len(self.plants_to_remove): # add to total system flowe
                    self.system_flows['total system flow'] = subset_system_flow
                else: # add to subset column
                    self.system_flows[', '.join(subset)] = subset_system_flow  
            
            # Save system flows of all subsets to excel
            self.system_flows.to_csv(parent_folder / "predictions" / f"System {mode} {self.scenario}.csv", index=True) 
        
        return self.system_flows
         
    def predict_internal_demand(self, data, subset):
        '''
        Calculates internal boiler steam demand for a subset

        Use pre-trained ML models to predict the swing boiler generation of a particular subset of plants for all time points
        '''
        # Modify data to only include plants in subset
        data["Total Steam Demand"] = data['CST_sum']  # reset summed steamd demand to rebuild later
        for player in self.plants_to_remove:  
            if player in subset:
                data["Total Steam Demand"] = data["Total Steam Demand"] + data[player] # add plant steam demand to summed steam demand
            else:
                data[player] = 0  # set steam demand from plant to 0
                if player.removesuffix("_sum") in ['CPM', 'PD', 'BPM', 'PM', 'HPM'] : # plants that have production rates
                        data[player.removesuffix("_sum") + ' On/Off'] = 0    # set production rate to 0
                
        data["Total Steam Demand--1"] = data["Total Steam Demand"].shift(1)     

        # Make predictions in t/hr
        predicted_internal_demand =  np.array(self.internal_demand_model.predict(data))
        return predicted_internal_demand
          
    def predict_swing_boiler_gen(self, data, subset, internal_demand):    
        '''
        Calculates swing boiler generation for a subset

        Use pre-trained ML models to predict the swing boiler generation of a particular subset of plants for all time points
        '''
        # Modify data to only include plants in subset
        data["Total Steam Demand"] = data['CST_sum'] + data['LP-EVP_sum'] # reset summed steamd demand to rebuild later
        #data['LP-S&R_sum'] = internal_demand # set internal demand to predicted   internal demand
        for player in self.plants_to_remove:  
                if player in subset: 
                    data["Total Steam Demand"] = data["Total Steam Demand"] + data[player] # add plant steam demand to summed steam demand
                    
                else:
                    data[player] = 0  # set steam demand from plant to 0
                    if player.removesuffix("_sum") in ['CPM', 'PD', 'BPM', 'PM', 'HPM'] : # plants that have production rates
                            data[player.removesuffix("_sum") + ' On/Off'] = 0    # set production rate to 0
                    
        data["Total Steam Demand--1"] = data["Total Steam Demand"].shift(1)
        data["BPM_sum--1"] = data["BPM_sum"].shift(1)
            
        # reove NaNs
        # Make predictions in t/hr
        predicted_swing_boiler_gen =  np.array(self.swing_boiler_model.predict(data))
        return predicted_swing_boiler_gen  
                
    def predict_system_costs(self, data, subset, internal_demand, swing_boiler_gen):
        '''
        Calculates system flow of a subset 

        Use pre-trained ML models to predict system cost of a particular subset of plants for all time points
        '''
        # Modify data to only include plants in subset
        data["Total Steam Demand"] = data['CST_sum'] + data['LP-EVP_sum'] # reset summed steamd demand to rebuild later
        #data['LP-S&R_sum'] = internal_demand # set internal demand to predicted   internal demand
        data['PB2 Steam Generation'] = swing_boiler_gen # set swing boiler generation to predicted swing boiler generation
        for player in self.plants_to_remove:  
                if player in subset: 
                    data["Total Steam Demand"] = data["Total Steam Demand"] + data[player] # add plant steam demand to summed steam demand
                    
                else:
                    data[player] = 0  # set steam demand from plant to 0
                    if player.removesuffix("_sum") in ['CPM', 'PD', 'BPM', 'PM', 'HPM'] : # plants that have production rates
                            data[player.removesuffix("_sum") + ' On/Off'] = 0    # set production rate to 0
                    
        # Predict fuel consumption in $/interval
        gas_consumption = np.array(self.fuel_model.predict(data)) # total gas consumption in NM3/hr
        subset_system_cost = gas_consumption*self.fuel_NHV/1000 * self.gas_cost * self.time_interval  
        
        
            
        return subset_system_cost
     
    def predict_system_revanue(self, data, subset): 
        '''
        Calculates system flow of a subset 

        Use pre-trained ML models to predict the revanue from turbine generation cost of a particular subset of plants for all time points
        '''
        # Modify data to only include plants in subset
        data["Total Steam Demand"] = data['CST_sum'] + data['LP-EVP_sum'] # reset summed steamd demand to rebuild later
       
        for player in self.plants_to_remove:  
                if player in subset: 
                    data["Total Steam Demand"] = data["Total Steam Demand"] + data[player] # add plant steam demand to summed steam demand
                    
                else:
                    data[player] = 0  # set steam demand from plant to 0
                    if player.removesuffix("_sum") in ['CPM', 'PD', 'BPM', 'PM', 'HPM'] : # plants that have production rates
                            pass
                            #data[player.removesuffix("_sum") + ' On/Off'] = 0    # set production rate to 0
        data["Total Steam Demand--1"] = data["Total Steam Demand"].shift(1)
                    
        # Predict turbine generation revanue in $/interval
        turbine_generation = np.array(self.turbine_model.predict(data)) # total turbine generation in MW
        subset_system_revanue = turbine_generation * self.elec_cost * self.time_interval # total system revenue in $/interval
            
        return subset_system_revanue
       
    def postprocess_results(self, results, mode):
        # Setup dataframes to store results
        steam_costs_df = pd.DataFrame(columns=self.plants_to_remove, index=self.fuel_data.index) # bulk steam costs for each plant i
        plant_allocations_df = pd.DataFrame(columns=self.plants_to_remove, index=self.fuel_data.index) # allocations for each plant i
        
        # Extract cost allocations from each time point
        i = 0
        for res_i in results:
            row_time, row_steam_costs, row_cost_allocations = res_i
            steam_costs_df.loc[row_time] = row_steam_costs
            plant_allocations_df.loc[row_time] = [row_cost_allocations[i].value[0] for i in range(len(row_cost_allocations))]   
            
            i+=1        

        # Calculate average steam cost and save to file
        average_steam_costs = self.steam_costs_df.mean(skipna=True).to_frame().transpose()
        average_steam_costs.columns = self.steam_costs_df.columns
        steam_costs_df.to_excel(parent_folder / "predictions" / f"Bulk Steam Costs {mode}.xlsx", index=True) # save matrix to excel
        plant_allocations_df.to_csv(parent_folder / "predictions" / f"Plant Allocations {mode} {self.scenario}.csv", index=True) # save matrix to excel
    
        print(plant_allocations_df)   
       
    def save_predictions(
        self,
        filename # name of initial data file to include in save name
        ):
        '''Save predictions to excel file'''

        # Remove the file format from the raw data file name
        sub_list = [".csv", ".xlsx"]
        for sub in sub_list: 
            name = filename.replace(sub , '') 

        # Save actual and prediction data 
        file_to_save = parent_folder / "outputs" / "predictions" / "{} Predictions {}.xlsx".format(self.target_col_name, name)  # CWD relative file path for input files
        self.predicted_df.to_excel(file_to_save, index=True) # save matrix to excel
        
def get_plant_allocations(row_time,
        row_system_flows,
        gas_cost,
        elec_cost,
        dH_steam,
        dH_turbine,
        time_interval,
        plants_to_remove,
        subset_combinations,
        row_steam_demands,
        mode,
        row_number
        ):
    '''
    Use game theory to allocate costs/revenues to plants

    '''
    
    # Print progress every 10,000 rows
    if row_number % 10_000 == 0:
        print(f"Processing row {row_number}: Time {row_time}...")
    
    # Convert the index into sets for fast lookup
    index_sets = {frozenset(key.split(", ")): value for key, value in row_system_flows.items()}
    lookup_cache = {}  # Dictionary to store cached results

    # Optimized function to lookup value by a set with caching
    def lookup_value(key_set):
        key = frozenset(key_set)  # Convert to frozenset for consistency
        if key in lookup_cache:  
            return lookup_cache[key]  # Return cached value if available

        result = index_sets.get(key, 0)  # Fetch from main dictionary
        lookup_cache[key] = result  # Store in cache
        return result  # Return the value


    # Precompute plant indices
    plant_indices = {plant: idx for idx, plant in enumerate(plants_to_remove)}
    
    # GEKKO Model
    m = GEKKO(remote=False)
    m.options.IMODE = 3
    m.options.SOLVER = 3

    # Compute marginal contributions
    marginal_contributions = np.zeros(len(plants_to_remove))

    for player in plants_to_remove:
        player_index = plant_indices[player]
        for subset in subset_combinations:
            if player in subset:
                subset_without_player = frozenset(set(subset) - {player})

                if len(subset) == 1:
                    system_cost_with_player = 0
                    system_cost_without_player = 0
                elif len(subset_without_player) == 1:
                    system_cost_with_player = lookup_value(subset)
                    system_cost_without_player = 0
                elif len(subset) == len(plants_to_remove):
                    system_cost_with_player = row_system_flows.get('total system flow', 0)
                    system_cost_without_player = lookup_value(subset_without_player)
                else:
                    system_cost_with_player = lookup_value(subset)
                    system_cost_without_player = lookup_value(subset_without_player)

                marginal_contributions[player_index] += float(system_cost_with_player - system_cost_without_player)

    # Get total system flow
    total_system_flow = max(row_system_flows.get('total system flow', 1e-6), 1e-6)

    # Compute plant weights using NumPy
    W_i = np.where(marginal_contributions != 0, marginal_contributions / total_system_flow, 0)

    # Variables
    if mode == 'Costs':
        X_i = [
            m.Var(
                value=0, 
                lb=0, 
                ub=row_steam_demands.get(plant, 0) * time_interval * 1000 * dH_steam / 1e6 / 0.8 * gas_cost) # cost if all steam came from gas boiler tuned for that load, in some cases by sharing a utility system we could get lower boiler efficiencies since its not tuned for just 1 demand
                if np.abs(marginal_contributions[plant_indices[plant]]) > 1 else m.Param(value=0)
            for plant in plants_to_remove
        ]
    else:
        X_i = [
            m.Var(
                value=0, 
                lb=0,
                ) # probs cant have upper bounds since its too small since there is a lot of steam not accounted for in these plants
                if np.abs(marginal_contributions[plant_indices[plant]]) > 1 else m.Param(value=0)
                for plant in plants_to_remove
        ]

    Z = m.Var(value=0)

    # Constraints. Note that the constraint forcing the X_i to be better than what it could get by itself is included as a bound on the variabe. For costs the cost of i must be lower than all gas steam and for revanue it must be above 0 (no turbine)
    m.Equation(sum(X_i) == total_system_flow)
    for i in range(len(plants_to_remove)):
        if np.abs(marginal_contributions[i]) > 1:
            m.Equation(Z <= X_i[i] / W_i[i])
    
    # Objective Function
    m.Maximize(Z)

    # Solve
    try:
        m.solve(disp=False)
        mSuccess = 1
    except:
        mSuccess = 0
        print(f'Failed to solve at time {row_time}')

    # Cleanup
    m.cleanup()

    steam_costs = [np.nan] * len(plants_to_remove)
    return row_time, steam_costs, X_i
  
