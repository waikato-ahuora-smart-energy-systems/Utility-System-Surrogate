from pathlib import Path
import pandas as pd

from scripts import Data, Learn, Correlate, SteadyState, Validate
import numpy as np
import copy
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import math
from sklearn.preprocessing import LabelEncoder
import pickle
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import CoolProp.CoolProp as CP  
 
def train_EV1():
    # User inputs
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    train_data_filename = 'EV1 Aug21-Dec24.csv' # 
    target_name = '450 STEAM TO EV1 EFFECT 1' 
    cols_to_keep = ['450 STEAM TO EV1 EFFECT 1', # steam flow to evaps t/hr
                    'EV1 Out Solids %',  # outlet solids %
                    'EV1 In Liquor FLR', # instantenous flowrate to evaps L/min
                    'EV1 In Liquor Temp', # WBL inlet temperature oC
                    'EV1 In Solids %', # instantenous inlet solids %
                    'Water Temp', # cooling water temperature oC
                    'CPM Rate', 'BPM Rate', 'HPM Rate', # CPM Rate ADT/day 
                    'EV1 Vacuum Pressure', # surface condenser vacuum kPa abs
                    'Shift Letter' # shift letter
                    ]
    param_dict =  {'n_estimators': 900, 'learning_rate': np.float64(0.07367381580193479), 'max_depth': 10, 'min_child_weight': np.float64(7.0), 'subsample': np.float64(0.6930739011300584), 'colsample_bytree': np.float64(0.8412517421186451), 'gamma': np.float64(3.276988651222436), 'lambda': np.float64(6.834729715489212), 'alpha': np.float64(3.619936864552426)}
    scaler = None
    
    ## Data import ##
    raw_file_to_open = parent_folder / "data" / train_data_filename  # CWD relative file path for input files
    raw = Data(target_col_name = target_name)
    raw.data_import(file_to_open = raw_file_to_open, header_row=2, data_start_row=19, time_col_name='Descr') # This needs to be commented out when filtered_filename on data.py is uncommented
    
    ## Feature engineering ##
    raw.raw = raw.raw[cols_to_keep]

    # Add Gaussian noise to solids column since data resolution is lower than others
    raw.raw['EV1 In Solids %'] += np.random.normal(loc=0, scale=0.2, size=len(raw.raw['EV1 In Solids %']))
    raw.raw['EV1 Out Solids %'] += np.random.normal(loc=0, scale=0.2, size=len(raw.raw['EV1 Out Solids %']))
    
    # Encode the letters in the 'Shift Letter' column to numbers
    encoder = LabelEncoder()
    raw.raw['Shift Number'] = encoder.fit_transform(raw.raw['Shift Letter'])
    raw.raw.drop(columns='Shift Letter', inplace=True)
    
    # Add boilout counter
    counter = 0
    counters = []
    for val in raw.raw['EV1 In Liquor Temp']:
        if val > 55: # temperature below 55oC indicates warm water boilout
            counter += 1
        else:
            counter = 0
        counters.append(counter)
    raw.raw['Time Since Boilout'] = counters # number of timepoints since boilout
    raw.raw['Time Since Boilout'] = raw.raw['Time Since Boilout'] / 12 # number of hours since boilout
 
    
    # Create lagged features for data that is highly dynamic and has a time delay
    raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='EV1 In Liquor FLR', lag_indices=[-2], step=1)
    raw.raw['Rolling Inlet Solids'] = raw.raw['EV1 In Solids %'].rolling(window=5).mean()
    raw.raw['Rolling Water Temp'] = raw.raw['Water Temp'].rolling(window=5).mean()
      
    # Smooth the target
    raw.raw[target_name] = savgol_filter(raw.raw[target_name], window_length=24, polyorder=3) 
 
    ## Inspect data ##
    raw.inspect_data(raw.raw)

    ## Filter data ##
    steady = SteadyState(data=raw.raw, rolling_width=12, pct=70, cols_to_steady=['450 STEAM TO EV1 EFFECT 1',]) # Steady state
    raw.raw = steady.steadied
    raw.filter(filter_cols=None)
    raw.filtered = raw.filtered[raw.filtered['EV1 In Liquor Temp'] > 55] # remove boilouts evident by a temperature around 50oC as its using warm water
    raw.filtered = raw.filtered[raw.filtered['450 STEAM TO EV1 EFFECT 1'] > 10] # remove periods of shutdown
    
    ## Train model ##
    data = raw.filtered
    
    # Split training (train, tuning and testing is included) and validation data
    val_split_start_date = '3/09/2023 00:00:00' #'1/08/2023 00:00:00'  #'3/09/2023 00:00:00'
    val_split_end_date = '20/09/2023 00:00:00' #'1/01/2025 00:00:00' #'20/09/2023 00:00:00'
    
    training_data = data[(data.index >= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S')) & (data.index < pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S'))]
    validation_data = data[(data.index >= pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S')) & (data.index <= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S'))]
    
    # Scale data
    scaler_X = None # StandardScaler() # RobustScaler() StandardScaler()  MinMaxScaler()
    scaler_Y = None
    
    if scaler_X is not None:
        X = training_data.drop(columns=[target_name])  # Exclude the target column
        scaler_X.fit(X)
        X_scaled = scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=training_data.index, columns=X.columns)
        
        Y = training_data[target_name]
        Y_scaled = Y
        #scaler_Y.fit(Y.values.reshape(-1, 1))
        #Y_scaled =  scaler_Y.transform(Y.values.reshape(-1, 1))
        training_data.loc[:, target_name] = Y_scaled 
    
    from_pickle = False
    if from_pickle: # no need to train new model, go straight to predictions
        pass
    else: # train new model 
        # Train ML model 
        train_data = copy.deepcopy(training_data) # make copy of data to train model on
        
        # Correlation analysis
        correlate = Correlate(data=train_data, target_col_name=target_name)
        correlate.correlate_rates(feature_names=train_data.columns)
        correlate.importance(param_dict=param_dict, top_n=30)
        correlate.importance_plot()
        #train_data = correlate.filtered # option to retain only the most important features
        plt.show()
        
        training = Learn(train_data=train_data, target_col_name=target_name, weight_method='time*density', c_log=500, a_weight=0.8, scaler_X=scaler_X, scaler_Y=scaler_Y)
        training.train(param_dict=param_dict, hyper_tune=False)
        training.pickle_model()
        plt.show()
       
   
    ## Validate model ##
    validation = Validate(test_data=validation_data, model_name=target_name, target_col_name=target_name, scaler_X=scaler_X, scaler_Y=scaler_Y)
    validation.validate_previous()
       
def train_EV2():     
    # User inputs
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    train_data_filename = 'No5 Evaps Aug21-Dec24.csv'
    target_name = 'EV2 STEAM TOTAL'   
    cols_to_keep = ['EV2 STEAM TOTAL', # steam flow to evaps t/hr
                    'EV2 Out Solids %', # EV2 Out Solids % %
                    'EV2 In Liquor FLR',	# inlet WBL flowrate L/min
                    'Internal Sweetning', # Internal SweetniPB2 NG Flowrate from HBL L/min
                    'EV2 In Solids %', # WBL inlet solids %
                    'EV2 In Liquor Temp',	# WBL inlet temperature oC
                    'Air Temp', # air temperature (since connected to cooling tower)
                    'Wash Liquor FLR', # flowrate of Wash Liquor FLR L/min
                    'EV2 Vacuum Pressure', # vacuum pressure kPa abs
                    'Shift Letter', # shift letter
                    'CPM Rate', 'BPM Rate', 'HPM Rate', #CPM, 1 mill & HPM CPM Rate ADT/day 
                    ]
    
    param_dict = {'n_estimators': 400, 'learning_rate': np.float64(0.02967215853826168), 'max_depth': 6, 'subsample': np.float64(0.6345222746098701), 'colsample_bytree': np.float64(0.9839123448080073), 'lambda': np.float64(4.476256096398036)}
    
    ## Data import ##
    raw_file_to_open = parent_folder / "data" / train_data_filename  # CWD relative file path for input files
    raw = Data(target_col_name = target_name)
    raw.data_import(file_to_open = raw_file_to_open, header_row=2, data_start_row=19, time_col_name='Descr') # This needs to be commented out when filtered_filename on data.py is uncommented
    
    ## Feature engineering ##
    raw.raw = raw.raw[cols_to_keep]

    # Add Gaussian noise to solids column
    raw.raw['EV2 In Solids %'] += np.random.normal(loc=0, scale=0.2, size=len(raw.raw['EV2 In Solids %']))
    
    # Create lagged features for data that is highly dynamic and has a time delay
    raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='EV2 In Liquor FLR', lag_indices=[-1], step=1)
    
    # Encode the letters in the 'Shift Letter' column to numbers
    encoder = LabelEncoder()
    raw.raw['CURRENT SHIFT Number'] = encoder.fit_transform(raw.raw['Shift Letter'])
    raw.raw.drop(columns='Shift Letter', inplace=True)
    
    # Smooth the target
    raw.raw[target_name] = savgol_filter(raw.raw[target_name], window_length=24, polyorder=3) 
   
    ## Filter data ##
    raw.filter(filter_cols=None)
    raw.filtered = raw.filtered[raw.filtered['EV2 In Liquor Temp'] > 60] # remove boilouts evident by a temperature around 50oC as its using warm water
    raw.filtered = raw.filtered[raw.filtered['EV2 Out Solids %'] > 60] # remove periods with extreme outlet low solids
    
    ## Train model ##
    data = raw.filtered

    # Split training (train, tuning and testing is included) and validation data
    val_split_start_date = '3/09/2023 00:00:00' #'1/08/2023 00:00:00'  #'3/09/2023 00:00:00'
    val_split_end_date = '20/09/2023 00:00:00' #'1/01/2025 00:00:00' #'20/09/2023 00:00:00'
    
    training_data = data[(data.index >= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S')) & (data.index < pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S'))]
    validation_data = data[(data.index >= pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S')) & (data.index <= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S'))]
    
    # Scale data
    scaler_X = None # StandardScaler() # RobustScaler() StandardScaler()  MinMaxScaler()
    scaler_Y = None
    
    if scaler_X is not None:
        X = training_data.drop(columns=[target_name])  # Exclude the target column
        scaler_X.fit(X)
        X_scaled = scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=training_data.index, columns=X.columns)
        
        Y = training_data[target_name]
        Y_scaled = Y
        #scaler_Y.fit(Y.values.reshape(-1, 1))
        #Y_scaled =  scaler_Y.transform(Y.values.reshape(-1, 1))
        training_data.loc[:, target_name] = Y_scaled
        
    from_pickle = False
    if from_pickle: # no need to train new model, go straight to predictions
        pass
    else: # train new model 
        # Train ML model 
        train_data = copy.deepcopy(training_data) # make copy of data to train model on
        
        # Correlation analysis
        correlate = Correlate(data=train_data, target_col_name=target_name)
        correlate.correlate_rates(feature_names=train_data.columns)
        correlate.importance(param_dict=param_dict, top_n=30)
        correlate.importance_plot()
        #train_data = correlate.filtered # option to retain only the most important features
        plt.show()
        
        training = Learn(train_data=train_data, target_col_name=target_name, weight_method='time*density', c_log=500, a_weight=0.8, scaler_X=scaler_X, scaler_Y=scaler_Y)
        training.train(param_dict=param_dict, hyper_tune=False)
        training.pickle_model()
        plt.show()
           
    
    ## Validate model ##
    validation = Validate(test_data=validation_data, model_name=target_name, target_col_name=target_name, scaler_X=scaler_X, scaler_Y=scaler_Y)
    validation.validate_previous()
    
def train_RB1(): 
    ## User inputs ##
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    train_data_filename = 'RB1 Aug21-Dec24.csv'
    target_name = 'RB1 Steam Generation'   
    cols_to_keep = ['RB1 Steam Generation', 'Natural Gas to RB1', 'RB1 Liquor FLR',	'RB1 Liquor Solids %',	'Air Temp',
                    'IP-PM', 'MP-PM', 
                    'MP-PD', 
                    'MP-BPM', 
                    'MP-CPM',	'MP-BLP', 
                    'MP-HPM',  
                    'LP-CPM', 'LP-BLP',	'LP-CST', 'LP-CEP', 'LP-EVP'
                    ]  

    scaler = None
    param_dict = {'n_estimators': 100, 'learning_rate': np.float64(0.05072815280932072), 'max_depth': 6, 'subsample': np.float64(0.886259566915398), 'colsample_bytree': np.float64(0.9705422281237182), 'lambda': np.float64(4.772321006665333)}
    raw_file_to_open = parent_folder / "data" / train_data_filename  # CWD relative file path for input files
    raw = Data(target_col_name = target_name)
    raw.data_import(file_to_open = raw_file_to_open, header_row=2, data_start_row=19, time_col_name='Descr') # This needs to be commented out when filtered_filename on data.py is uncommented
    
    ## Feature engineering ##
    raw.raw = raw.raw[cols_to_keep]   
    raw.raw['Rolling Temp'] = raw.raw['Air Temp'].rolling(window=5).mean()
    raw.raw['Gas On/Off'] = np.where(raw.raw['Natural Gas to RB1'] > 5, 1, 0) # binary signal for gas on/off
    
    # Sum steam demand data for each plant into a single group
    plant_groups = {
        'LP-EVP_sum': [ 'LP-EVP'], #'HP-EVP', 'MP-EVP',
        'PM_sum': ['IP-PM', 'MP-PM'],
        'PD_sum': ['MP-PD'],
        'mill1_sum': ['MP-BPM'],
        'CD2_sum': ['MP-CPM', 'LP-CPM'],
        'BLP_sum': ['MP-BLP', 'LP-BLP'],
        'CD3_sum': ['MP-HPM'],
        'C_K_sum': ['LP-CST'],
        'CEP_sum': ['LP-CEP']
    }

    for group, columns in plant_groups.items():
        raw.raw[group] = raw.raw[columns].sum(axis=1) # sum the steam demands for each plant group
        raw.raw = raw.raw.drop(columns=columns, axis=1) # drop original steam demands
    raw.raw['Total Steam Demand'] = raw.raw[plant_groups.keys()].sum(axis=1) 

    # Smooth the target
    raw.raw[target_name] = savgol_filter(raw.raw[target_name], window_length=50, polyorder=3) 
   
    ## Filter data ##
    steady = SteadyState(data=raw.raw, rolling_width=6, pct=70, cols_to_steady=['RB1 Steam Generation',]) # Steady state
    raw.raw = steady.steadied
    raw.filter(filter_cols=None)
    raw.filtered = raw.filtered[raw.filtered['RB1 Steam Generation'] > 50] # remove periods of effective shutdown
    raw.filtered.drop(columns=['Natural Gas to RB1'], inplace=True)    
    
    ## Train model ##
    data = raw.filtered

    # Split training (train, tuning and testing is included) and validation data
    val_split_start_date = '3/09/2023 00:00:00' #'1/08/2023 00:00:00'  #'3/09/2023 00:00:00'
    val_split_end_date = '20/09/2023 00:00:00' #'1/01/2025 00:00:00' #'20/09/2023 00:00:00'
    
    training_data = data[(data.index >= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S')) & (data.index < pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S'))]
    validation_data = data[(data.index >= pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S')) & (data.index <= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S'))]
    
    # Scale data
    scaler_X = None # StandardScaler() # RobustScaler() StandardScaler()  MinMaxScaler()
    scaler_Y = None
    
    if scaler_X is not None:
        X = training_data.drop(columns=[target_name])  # Exclude the target column
        scaler_X.fit(X)
        X_scaled = scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=training_data.index, columns=X.columns)
        
        Y = training_data[target_name]
        Y_scaled = Y
        #scaler_Y.fit(Y.values.reshape(-1, 1))
        #Y_scaled =  scaler_Y.transform(Y.values.reshape(-1, 1))
        training_data.loc[:, target_name] = Y_scaled
    
    from_pickle = False
    if from_pickle: # no need to train new model, go straight to predictions
        pass
    else: # train new model 
        # Train ML model 
        train_data = copy.deepcopy(training_data) # make copy of data to train model on
        

        # Correlation analysis
        correlate = Correlate(data=train_data, target_col_name=target_name)
        correlate.correlate_rates(feature_names=train_data.columns)
        correlate.importance(param_dict=param_dict, top_n=30)
        correlate.importance_plot()
        #train_data = correlate.filtered # option to retain only the most important features
        plt.show()
        
        training = Learn(train_data=train_data, target_col_name=target_name, weight_method='time*density', c_log=650, a_weight=0.8, scaler_X=scaler_X, scaler_Y=scaler_Y)
        training.train(param_dict=param_dict, hyper_tune=False)
    
        training.pickle_model()
        
    ## Validate model ##
    validation = Validate(test_data=validation_data, model_name=target_name, target_col_name=target_name, scaler_X=scaler_X, scaler_Y=scaler_Y)
    validation.validate_previous()
        
def train_RB2(): 
    ## User inputs ##
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    train_data_filename = 'RB2 Aug21-Dec24.csv'
    target_name = 'RB2 Steam Generation'  

    cols_to_keep = ['RB2 Steam Generation', 'Gas to RB2',	'RB2 Ignition Gas', 'RB2 Liquor FLR', 'RB2 Liquor Solids %', 'Air Temp',
                    'IP-PM', 'MP-PM', 
                    'MP-PD', 
                    'MP-BPM', 
                    'MP-CPM',	'MP-BLP', 
                    'MP-HPM',  
                    'LP-CPM', 'LP-BLP',	'LP-CST', 'LP-CEP', 'LP-EVP'
                    ]  
    
    param_dict = {'n_estimators': 250, 'learning_rate': np.float64(0.1466086202922278), 'max_depth': 5, 'subsample': np.float64(0.8935701876536052), 'colsample_bytree': np.float64(0.7641030483339484), 'lambda': np.float64(1.0167246085184165)}
    raw_file_to_open = parent_folder / "data" / train_data_filename  # CWD relative file path for input files
    raw = Data(target_col_name = target_name)
    raw.data_import(file_to_open = raw_file_to_open, header_row=2, data_start_row=19, time_col_name='Descr') # This needs to be commented out when filtered_filename on data.py is uncommented
    
    ## Feature engineering ##
    raw.raw = raw.raw[cols_to_keep]   
    raw.raw['RB2 Gas Total'] = raw.raw['Gas to RB2'] + raw.raw['RB2 Ignition Gas']
    raw.raw['Rolling Temp'] = raw.raw['Air Temp'].rolling(window=5).mean()
    raw.raw['Gas On/Off'] = np.where(raw.raw['RB2 Gas Total'] > 25, 1, 0)
    
    # Smooth the target
    raw.raw[target_name] = savgol_filter(raw.raw[target_name], window_length=50, polyorder=3) 
    
    # Sum steam demand data for each plant into a single group
    plant_groups = {
        'LP-EVP_sum': [ 'LP-EVP'], 
        'PM_sum': ['IP-PM', 'MP-PM'],
        'PD_sum': ['MP-PD'],
        'mill1_sum': ['MP-BPM'],
        'CD2_sum': ['MP-CPM', 'LP-CPM'],
        'BLP_sum': ['MP-BLP', 'LP-BLP'],
        'CD3_sum': ['MP-HPM'],
        'C_K_sum': ['LP-CST'],
        'CEP_sum': ['LP-CEP']
    }

    for group, columns in plant_groups.items():
        raw.raw[group] = raw.raw[columns].sum(axis=1) # sum the steam demands for each plant group
        raw.raw = raw.raw.drop(columns=columns, axis=1) # drop original steam demands
    raw.raw['Total Steam Demand'] = raw.raw[plant_groups.keys()].sum(axis=1) 

    #raw.raw['Total Steam Demand'+'_ramp_rate'] = raw.apply_savgol_filter(data = raw.raw['Total Steam Demand'], window_length=6, polyorder=3)[1] 
    #raw.raw = raw.raw.drop(columns='Total Steam Demand'+'_ramp_rate', axis=1)
    
    ## Filter data ##
    steady = SteadyState(data=raw.raw, rolling_width=6, pct=70, cols_to_steady=['RB2 Steam Generation',]) # Steady state
    raw.raw = steady.steadied
    raw.filter(filter_cols=None)
    raw.filtered = raw.filtered[raw.filtered['RB2 Steam Generation'] > 20]
    #raw.save_processed(filename=train_data_filename)
    raw.filtered.drop(columns=['RB2 Gas Total', 'Gas to RB2', 'RB2 Ignition Gas'], inplace=True)    

    ## Train model ##
    # Define date boundaries
    data = raw.filtered

    # Split training (train, tuning and testing is included) and validation data
    val_split_start_date = '3/09/2023 00:00:00' #'1/08/2023 00:00:00'  #'3/09/2023 00:00:00'
    val_split_end_date = '20/09/2023 00:00:00' #'1/01/2025 00:00:00' #'20/09/2023 00:00:00'
    
    training_data = data[(data.index >= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S')) & (data.index < pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S'))]
    validation_data = data[(data.index >= pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S')) & (data.index <= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S'))]
    
    # Scale data
    scaler_X = None # StandardScaler() # RobustScaler() StandardScaler()  MinMaxScaler()
    scaler_Y = None
    
    if scaler_X is not None:
        X = training_data.drop(columns=[target_name])  # Exclude the target column
        scaler_X.fit(X)
        X_scaled = scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=training_data.index, columns=X.columns)
        
        Y = training_data[target_name]
        Y_scaled = Y
        #scaler_Y.fit(Y.values.reshape(-1, 1))
        #Y_scaled =  scaler_Y.transform(Y.values.reshape(-1, 1))
        training_data.loc[:, target_name] = Y_scaled
    
    from_pickle = False
    if from_pickle: # no need to train new model, go straight to predictions
        pass
    else: # train new model 
        # Train ML model 
        train_data = copy.deepcopy(training_data) # make copy of data to train model on
        
        # Correlation analysis
        correlate = Correlate(data=train_data, target_col_name=target_name)
        correlate.correlate_rates(feature_names=train_data.columns)
        correlate.importance(param_dict=param_dict, top_n=30)
        correlate.importance_plot()
        #train_data = correlate.filtered # option to retain only the most important features
        plt.show()
        
        
        training = Learn(train_data=train_data, target_col_name=target_name, weight_method='time*density', c_log=500, a_weight=0.8, scaler_X=scaler_X, scaler_Y=scaler_Y)
        training.train(param_dict=param_dict, hyper_tune=False )
        #training.pickle_model()
        plt.show()
       
    ## Validate model ##
    validation = Validate(test_data=validation_data, model_name=target_name, target_col_name=target_name, scaler_X=scaler_X, scaler_Y=scaler_Y)
    validation.validate_previous()

def train_turbine(process_raw_data = True, filter_data=True, train_new_model = True, validate_model=False):
    ## User inputs ##
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    train_data_filename = 'Steam Generation Aug21-Dec24.csv'
    target_name = 'Turbine Power Generation'   
    cols_to_keep = ['Turbine Power Generation',
                    'Vent FLR', 
                    'RB1 Liquor FLR', 'RB2 Liquor FLR', # RB1 & RB2 liquor flowrate L/min
                    'RB1 Liquor Solids %', 'RB2 Liquor Solids %', # RB1 & RB2 solids %
                    'Air Temp',  #'Air Temp',
                    'IP-PM', 'MP-PM', 
                    'MP-PD', 
                    'MP-BPM', 
                    'MP-CPM',	'MP-BLP', 
                    'MP-HPM',  
                    'LP-CPM', 'LP-BLP',	'LP-CST', 'LP-CEP', 
                    'LP-EVP', 
                    ]  
    
    param_dict  = {'n_estimators': 600, 'learning_rate': np.float64(0.02068861661030041), 'max_depth': 10, 'subsample': np.float64(0.8629607809467031), 'colsample_bytree': np.float64(0.637820083656556), 'lambda': np.float64(2.0792537871453036)}
  
    if process_raw_data: # import raw data & process
        raw_file_to_open = parent_folder / "data" / train_data_filename  # CWD relative file path for input files
        raw = Data(target_col_name = target_name)
        raw.data_import(file_to_open = raw_file_to_open, header_row=2, data_start_row=19, time_col_name='Descr') # This needs to be commented out when filtered_filename on data.py is uncommented
        
        ## Feature engineering ##
        raw.raw = raw.raw[cols_to_keep]   
                
        # Sum steam demand data for each plant into a single group
        plant_groups = {
            'LP-EVP_sum': ['LP-EVP'], 
            'PM_sum': ['IP-PM', 'MP-PM'],
            'PD_sum': ['MP-PD'],
            'mill1_sum': ['MP-BPM'],
            'CD2_sum': ['MP-CPM', 'LP-CPM'],
            'BLP_sum': ['MP-BLP', 'LP-BLP'],
            'CD3_sum': ['MP-HPM'],
            'C_K_sum': ['LP-CST'],
            'CEP_sum': ['LP-CEP']
        }
     
        for group, columns in plant_groups.items():
            raw.raw[group] = raw.raw[columns].sum(axis=1) # sum the steam demands for each plant group
            #raw.raw = raw.raw.drop(columns=columns, axis=1) # drop original steam demands
        raw.raw['Total Steam Demand'] = raw.raw[plant_groups.keys()].sum(axis=1) 
        
        for group, columns in plant_groups.items():
            raw.raw.drop(columns=group)
        
        # Rolling values
        raw.raw['Rolling Temp FD'] = raw.raw['Air Temp'].rolling(window=5).mean()
        raw.raw.drop(columns=['Air Temp'], inplace=True)
        raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='Summed Steam Demand', lag_indices=[-1], step=1)
        
        # TDS of Liquor
        raw.raw['RB1 TDS/d'] = (raw.raw['RB1 Liquor FLR']/1000/60*(582.68*raw.raw['RB1 Liquor Solids %']/100+958.86)*raw.raw['RB1 Liquor Solids %']/100)/1000*3600*24
        raw.raw['RB2 TDS/d'] = (raw.raw['RB2 Liquor FLR']/1000/60*(582.68*raw.raw['RB2 Liquor Solids %']/100+958.86)*raw.raw['RB2 Liquor Solids %']/100)/1000*3600*24
        raw.raw.drop(columns=['RB1 Liquor FLR', 'RB1 Liquor Solids %', 'RB2 Liquor FLR', 'RB2 Liquor Solids %'], inplace=True) 
        
        if filter_data: 
            ## Filter data ##
            steady = SteadyState(data=raw.raw, rolling_width=24, pct=80, cols_to_steady=[target_name,]) # Steady state
            raw.raw = steady.steadied
            raw.filter(filter_cols=None)
            raw.filtered = raw.filtered[raw.filtered[target_name] >= 10] # filter out periods of shutdown
            
            data = raw.filtered
        else: 
            data = raw.raw
           
    else: # Import pre-processed file to save time (this means that the lines of code for raw has already been completed)
        processed_filename = 'Processed {}.csv'.format(target_name.replace('.csv', ''))   
        processed_file_to_open = parent_folder / "data"  / processed_filename  # CWD relative file path for input files
        processed = Data()
        processed.data_import(file_to_open = processed_file_to_open,  header_row=1, data_start_row=2, time_col_name='Descr')
        data = processed.raw 
    
    ## Train model ##
    # Define date boundaries
    data = raw.filtered

    # Split training (train, tuning and testing is included) and validation data
    val_split_start_date = '3/09/2023 00:00:00' #'1/08/2023 00:00:00'  #'3/09/2023 00:00:00'
    val_split_end_date = '20/09/2023 00:00:00' #'1/01/2025 00:00:00' #'20/09/2023 00:00:00'
    
    training_data = data[(data.index >= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S')) & (data.index < pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S'))]
    validation_data = data[(data.index >= pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S')) & (data.index <= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S'))]
    
    # Scale data
    scaler_X = None # StandardScaler() # RobustScaler() StandardScaler()  MinMaxScaler()
    scaler_Y = None
    
    if scaler_X is not None:
        X = training_data.drop(columns=[target_name])  # Exclude the target column
        scaler_X.fit(X)
        X_scaled = scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=training_data.index, columns=X.columns)
        
        Y = training_data[target_name]
        Y_scaled = Y
        #scaler_Y.fit(Y.values.reshape(-1, 1))
        #Y_scaled =  scaler_Y.transform(Y.values.reshape(-1, 1))
        training_data.loc[:, target_name] = Y_scaled
    
    from_pickle = False
    if train_new_model: # train new model 
        # Train ML model 
        train_data = copy.deepcopy(training_data) # make copy of data to train model on
        
        # Correlation analysis
        correlate = Correlate(data=train_data, target_col_name=target_name)
        correlate.correlate_rates(feature_names=train_data.columns)
        correlate.importance(param_dict=param_dict, top_n=30)
        correlate.importance_plot()
        #train_data = correlate.filtered # option to retain only the most important features
        plt.show()
        
        training = Learn(train_data=train_data, target_col_name=target_name, weight_method='density', c_log=800, a_weight=1, scaler_X=scaler_X, scaler_Y=scaler_Y)
        training.train(param_dict=param_dict, hyper_tune=False)
        training.pickle_model()
        plt.show()
    
    if validate_model:
        ## Validate model ##
        validation = Validate(test_data=validation_data, model_name=target_name, target_col_name=target_name, scaler_X=scaler_X, scaler_Y=scaler_Y)
        validation.validate_previous()
        print(data.columns)
    return data
  
def train_natural_gas(process_raw_data = True, filter_data=True, train_new_model = True, validate_model=False):
    ## User inputs ##
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    train_data_filename = 'Steam Generation Aug21-Dec24.csv'
    target_name = 'Total NG' 
    cols_to_keep = ['IP-PM', 'MP-PM', 
                    'MP-PD', 
                    'MP-BPM', 
                    'MP-CPM',	'MP-BLP', 
                    'MP-HPM',  
                    'LP-CPM', 'LP-BLP',	'LP-CST', 'LP-CEP', 
                    'LP-EVP', 
                    'PB2 NG Flow', # PB2 gas flow NM3/hr
                    'RB1 Liquor Solids %', 'RB2 Liquor Solids %', # RB1 & RB2 solids %
                    'WW Moisture %', # WW moisture content %
                    'CPM Rate', 'PD Rate', 'BPM Rate', 'PM Rate',  'HPM Rate', # CPM Rates ADT/d
                    'RB1 Liquor FLR', 'RB2 Liquor FLR', # RB1, RB2 Liquor flow L/min
                    'Air Temp', # air temperature from outside oC
                    'Total NG', # total gas flow NM3/hr
                    'WW Screw Ratio', #  WW screw feeder upper ratio
                    'Midnight Rainfall', # rainfall since midnight mm
                    'Shift Letter', # shift number
                    'Vent FLR', # vent flowrate on LP header
                    'PB2 Steam Generation',	'RB1 Steam Generation',	'RB2 Steam Generation',	
                    ]  

    scaler = None   
    param_dict = {'n_estimators': 300, 'learning_rate': 0.05155605120993856, 'max_depth': 8, 'subsample': 0.7000000000000001, 'colsample_bytree': 0.6000000000000001, 'lambda': 3.1464313111556415} # all data
 
    if process_raw_data: # import raw data & process
        raw_file_to_open = parent_folder / "data" / train_data_filename  # CWD relative file path for input files
        raw = Data(target_col_name = target_name)
        raw.data_import(file_to_open = raw_file_to_open, header_row=2, data_start_row=19, time_col_name='Descr') # This needs to be commented out when filtered_filename on data.py is uncommented
        
        ## Feature engineering ##
        raw.raw = raw.raw[cols_to_keep]   
        
        # Encode the letters in the 'Shift Letter' column to numbers
        encoder = LabelEncoder()
        raw.raw['CURRENT SHIFT Number'] = encoder.fit_transform(raw.raw['Shift Letter'])
        raw.raw.drop(columns='Shift Letter', inplace=True)
        
        plant_groups = {
            'LP-EVP_sum': ['LP-EVP'],
            'PM_sum': ['IP-PM', 'MP-PM'],
            'PD_sum': ['MP-PD'],
            'mill1_sum': ['MP-BPM'],
            'CD2_sum': ['MP-CPM', 'LP-CPM'],
            'BLP_sum': ['MP-BLP', 'LP-BLP'],
            'CD3_sum': ['MP-HPM'],
            'C_K_sum': ['LP-CST'],
            'CEP_sum': ['LP-CEP']
        }

        # Sum steam demand data for each plant into a single group
        for group, columns in plant_groups.items():
            raw.raw[group] = raw.raw[columns].sum(axis=1)

        # Sum all grouped columns to get total steam demand
        summed_columns = list(plant_groups.keys())
        raw.raw['Total Steam Demand'] = raw.raw[summed_columns].sum(axis=1)

        # Drop intermediate group columns if no longer needed
        raw.raw.drop(columns=summed_columns, inplace=True)
        
        # Binary to denote plant on or off - this should help with categorising states since we dont want the model to learn steam demands when it sees a prod rate, but we know off/on is important
        prod_rate = ['CPM Rate', 'PD Rate', 'BPM Rate', 'PM Rate', 'HPM Rate']
        raw.raw['CPM Online'] = np.where(raw.raw['CPM Rate'] > 400, 1, 0)
        raw.raw['PD Online'] = np.where(raw.raw['PD Rate'] > 200, 1, 0)
        raw.raw['BPM Online'] = np.where(raw.raw['BPM Rate'] > 200, 1, 0)
        raw.raw['PM Online'] = np.where(raw.raw['PM Rate'] > 300, 1, 0)
        raw.raw['HPM Online'] = np.where(raw.raw['HPM Rate'] > 30, 1, 0)
        
        for prod in prod_rate:
            raw.raw.drop(columns=prod, inplace=True)
      
        # TDS of Liquor
        raw.raw['RB1 TDS/d'] = (raw.raw['RB1 Liquor FLR']/1000/60*(582.68*raw.raw['RB1 Liquor Solids %']/100+958.86)*raw.raw['RB1 Liquor Solids %']/100)/1000*3600*24
        raw.raw['RB2 TDS/d'] = (raw.raw['RB2 Liquor FLR']/1000/60*(582.68*raw.raw['RB2 Liquor Solids %']/100+958.86)*raw.raw['RB2 Liquor Solids %']/100)/1000*3600*24
        raw.raw.drop(columns=['RB1 Liquor FLR', 'RB1 Liquor Solids %', 'RB2 Liquor FLR', 'RB2 Liquor Solids %'], inplace=True) 
         
        # Time series features
        raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='Summed Steam Demand', lag_indices=[-1], step=1)
        raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='RB1 TDS/d', lag_indices=[-1], step=1)
        raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='RB2 TDS/d', lag_indices=[-1], step=1)
       
        # Cumulative Rainfall in past 24 hours instead of since midnight
        raw.raw['actual_rainfall'] = raw.raw['Midnight Rainfall'].diff().fillna(0)
        raw.raw['actual_rainfall'] = raw.raw['actual_rainfall'].clip(lower=0)
        raw.raw['Rainfall Sum'] = raw.raw['actual_rainfall'].rolling(window=288).sum()
        raw.raw.drop(columns=['actual_rainfall'], inplace=True)
        raw.raw.drop(columns=['Midnight Rainfall'], inplace=True)
       
        if filter_data: # Filter data
            ## Filter data ##
            steady = SteadyState(data=raw.raw, rolling_width=24, pct=70, cols_to_steady=[target_name,]) # Steady state
            raw.raw = steady.steadied
            raw.filter(filter_cols=None)
            
            raw.filtered = raw.filtered[(raw.filtered['PB2 NG Flow'] >= 900)] # remove periods below min gas
            raw.filtered = raw.filtered[(raw.filtered['Total NG'] <= 20000)] # remove extreme periods
            raw.filtered.drop(columns=['PB2 NG Flow'], inplace=True)
            
            # Remove any periods when any of the boilers were offline
            raw.filtered = raw.filtered[(raw.filtered['PB2 Steam Generation'] >= 5) & 
                                        (raw.filtered['RB1 Steam Generation'] >= 30) & 
                                        (raw.filtered['RB2 Steam Generation'] >= 30) #& 
                                        #(raw.filtered['2400 KPA STEAM FROM NO 7 PRI'] >= 10)
                                        ]
            
            #raw.filtered.drop(columns=['PB2 Steam Generation', ], inplace=True) #'2400 KPA STEAM FROM NO 7 PRI'
            raw.filtered.drop(columns=['RB1 Steam Generation', 'RB2 Steam Generation'], inplace=True) 
            data = raw.filtered
            raw.save_processed(filename=target_name+".csv") # save the processed data to a csv file since the feature engineering is time consuming
        else:
            raw.raw.drop(columns=['PB2 NG Flow', 'RB1 Steam Generation', 'RB2 Steam Generation'], inplace=True) # drop columns that were only used for filtering
            data = raw.raw
            
            
    else: # Import pre-processed file to save time (this means that the lines of code for raw has already been completed)
        processed_filename = 'Processed {}.csv'.format(target_name.replace('.csv', ''))   
        processed_file_to_open = parent_folder / "data"  / processed_filename  # CWD relative file path for input files
        processed = Data()
        processed.data_import(file_to_open = processed_file_to_open,  header_row=1, data_start_row=2, time_col_name='Descr')
        data = processed.raw 
    
    ## Train model ##
    # Define date boundaries
    data = raw.filtered

    # Split training (train, tuning and testing is included) and validation data
    val_split_start_date = '3/09/2023 00:00:00' #'1/08/2023 00:00:00'  #'3/09/2023 00:00:00'
    val_split_end_date = '20/09/2023 00:00:00' #'1/01/2025 00:00:00' #'20/09/2023 00:00:00'
    
    training_data = data[(data.index >= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S')) & (data.index < pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S'))]
    validation_data = data[(data.index >= pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S')) & (data.index <= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S'))]
    

    # Scale data
    scaler_X = None # StandardScaler() # RobustScaler() StandardScaler()  MinMaxScaler()
    scaler_Y = None # StandardScaler() 
    
    if scaler_X is not None:
        # X = training_data.drop(columns=[target_name])  # Exclude the target column
        # scaler_X.fit(X)
        # X_scaled = scaler_X.transform(X)
        # X_scaled_df = pd.DataFrame(X_scaled, index=training_data.index, columns=X.columns)
        
        Y = training_data[target_name]
        #Y_scaled = Y
        scaler_Y.fit(Y.values.reshape(-1, 1))
        Y_scaled =  scaler_Y.transform(Y.values.reshape(-1, 1))
        training_data.loc[:, target_name] = Y_scaled

    if train_new_model: # train new model
        # Train ML model 
        train_data = copy.deepcopy(training_data) # make copy of data to train model on
        
        # Correlation analysis
        correlate = Correlate(data=train_data, target_col_name=target_name)
        correlate.correlate_rates(feature_names=train_data.columns)
        correlate.importance(param_dict=param_dict, top_n=30)
        correlate.importance_plot()
        #train_data = correlate.filtered # option to retain only the most important features
        plt.show()
        
        training = Learn(train_data=train_data, target_col_name=target_name, weight_method='time*density', c_log=500, a_weight=0.4, scaler_X=scaler_X, scaler_Y=scaler_Y)
        training.train(param_dict=param_dict, hyper_tune=False)
        training.pickle_model()
        plt.show()
        
    if validate_model:  
        ## Validate model ##
        validation = Validate(test_data=validation_data, model_name=target_name, target_col_name=target_name, scaler_X=scaler_X, scaler_Y=scaler_Y)
        validation.validate_previous()
    return data

def train_PB2(process_raw_data = True, filter_data=True, train_new_model = True, validate_model=False): # boilers should have inputs relating to steam demand conditions whereas fuel/turbine models should have inputs relating to fuel conditions & boiler outputs so that we're not doubling up on inputs
    ## User inputs ##
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    train_data_filename = 'Steam Generation Aug21-Dec24.csv'
    target_name = 'PB2 Steam Generation'
    cols_to_keep = ['IP-PM', 'MP-PM', # don't need WW quality indicators since steam generation is not dependent since gas will be used as makeup if quality is poor
                    'MP-PD', 
                    'MP-BPM', 
                    'MP-CPM',	'MP-BLP', 
                    'MP-HPM',  
                    'LP-CPM', 'LP-BLP',	'LP-CST', 'LP-CEP', 
                    'LP-EVP', 
                    'RB1 Liquor Solids %', 'RB2 Liquor Solids %', # RB1 & RB2 solids %
                    'CPM Rate', 'PD Rate', 'BPM Rate', 'PM Rate',  'HPM Rate', # CPM Rates ADT/d
                    'RB1 Liquor FLR', 'RB2 Liquor FLR', # RB1, RB2 Liquor flow L/min
                    'Shift Letter', # shift number
                    'Vent FLR', # vent flowrate on LP header
                    'PB2 Steam Generation',	
                    'RB1 Steam Generation',	'RB2 Steam Generation',	
                    ]  
    
    param_dict = {'n_estimators': 800, 'learning_rate': np.float64(0.021638546177863663), 'max_depth': 12, 'subsample': np.float64(0.7000000000000001), 'colsample_bytree': np.float64(0.6000000000000001), 'lambda': np.float64(0.7161503408152412)}
    
    if process_raw_data: # import raw data & process
        raw_file_to_open = parent_folder / "data" / train_data_filename  # CWD relative file path for input files
        raw = Data(target_col_name = target_name)
        raw.data_import(file_to_open = raw_file_to_open, header_row=2, data_start_row=19, time_col_name='Descr') # This needs to be commented out when filtered_filename on data.py is uncommented
        
        ## Feature engineering ##
        raw.raw = raw.raw[cols_to_keep]   
        
        # Encode the letters in the 'Shift Letter' column to numbers
        encoder = LabelEncoder()
        raw.raw['CURRENT SHIFT Number'] = encoder.fit_transform(raw.raw['Shift Letter'])
        raw.raw.drop(columns='Shift Letter', inplace=True)
        
        plant_groups = {
            'LP-EVP-IDept_sum': ['LP-EVP'],
            'PM_sum': ['IP-PM', 'MP-PM'],
            'PD_sum': ['MP-PD'],
            'mill1_sum': ['MP-BPM'],
            'CD2_sum': ['MP-CPM', 'LP-CPM'],
            'BLP_sum': ['MP-BLP', 'LP-BLP'],
            'CD3_sum': ['MP-HPM'],
            'C_K_sum': ['LP-CST'],
            'CEP_sum': ['LP-CEP']
        }

        # Sum steam demand data for each plant into a single group
        for group, columns in plant_groups.items():
            raw.raw[group] = raw.raw[columns].sum(axis=1)

        # Sum all grouped columns to get total steam demand
        summed_columns = list(plant_groups.keys())
        raw.raw['Total Steam Demand'] = raw.raw[summed_columns].sum(axis=1)

        # Drop intermediate group columns if no longer needed
        raw.raw.drop(columns=summed_columns, inplace=True)
                
        # Binary to denote plant on or off - this should help with categorising states since we dont want the model to learn steam demands when it sees a prod rate, but we know off/on is important
        prod_rate = ['CPM Rate', 'PD Rate', 'BPM Rate', 'PM Rate', 'HPM Rate']
        raw.raw['CPM Online'] = np.where(raw.raw['CPM Rate'] > 400, 1, 0)
        raw.raw['PD Online'] = np.where(raw.raw['PD Rate'] > 200, 1, 0)
        raw.raw['BPM Online'] = np.where(raw.raw['BPM Rate'] > 200, 1, 0)
        raw.raw['PM Online'] = np.where(raw.raw['PM Rate'] > 300, 1, 0)
        raw.raw['HPM Online'] = np.where(raw.raw['HPM Rate'] > 30, 1, 0)
        
        for prod in prod_rate:
            raw.raw.drop(columns=prod, inplace=True)
        
        # TDS of Liquor
        raw.raw['RB1 TDS/d'] = (raw.raw['RB1 Liquor FLR']/1000/60*(582.68*raw.raw['RB1 Liquor Solids %']/100+958.86)*raw.raw['RB1 Liquor Solids %']/100)/1000*3600*24
        raw.raw['RB2 TDS/d'] = (raw.raw['RB2 Liquor FLR']/1000/60*(582.68*raw.raw['RB2 Liquor Solids %']/100+958.86)*raw.raw['RB2 Liquor Solids %']/100)/1000*3600*24
        raw.raw.drop(columns=['RB1 Liquor FLR', 'RB1 Liquor Solids %', 'RB2 Liquor FLR', 'RB2 Liquor Solids %'], inplace=True) 
         
        # Sum steam production from RB's
        raw.raw['RB Steam Prod'] = raw.raw['RB1 Steam Generation'] + raw.raw['RB2 Steam Generation'] # safer then using seperately since gas can be switched between them that the ML model doesnt pick up
        
        # Time series features
        raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='MP-BPM', lag_indices=[-1], step=1)
        raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='RB Steam Prod', lag_indices=[-1], step=1)
        raw.create_lagged_features(dataframe = raw.raw, lagged_col_name='Summed Steam Demand', lag_indices=[-1], step=1)
    
        if filter_data:
            ## Filter data ##
            steady = SteadyState(data=raw.raw, rolling_width=12, pct=70, cols_to_steady=[target_name,]) # Steady state
            raw.raw = steady.steadied
            raw.filter(filter_cols=None)
            
            # Remove periods where any of the boilers were offline
            raw.filtered = raw.filtered[(raw.filtered['PB2 Steam Generation'] >= 25) & 
                                        (raw.filtered['RB1 Steam Generation'] >= 30) & 
                                        (raw.filtered['RB2 Steam Generation'] >= 30) #& 
                                        #(raw.filtered['2400 KPA STEAM FROM NO 7 PRI'] >= 10)
                                        ]
            
            raw.filtered.drop(columns=['RB1 Steam Generation', 'RB2 Steam Generation'], inplace=True) 
            data = raw.filtered
            raw.save_processed(filename=target_name+'.csv') # save the processed data to a csv file since the feature engineering is time consuming
        else:
            raw.raw.drop(columns=['RB1 Steam Generation', 'RB2 Steam Generation'], inplace=True) # drop columns that were only used for filtering
            data = raw.raw
           
    else: # Import pre-processed file to save time (this means that the lines of code for raw has already been completed)
        processed_filename = 'Processed {}.csv'.format(target_name.replace('.csv', ''))   
        processed_file_to_open = parent_folder / "data"  / processed_filename  # CWD relative file path for input files
        processed = Data()
        processed.data_import(file_to_open = processed_file_to_open,  header_row=1, data_start_row=2, time_col_name='Descr')
        data = processed.raw 
    
    ## Train model ##
    # Define date boundaries
    data = raw.filtered

    # Split training (train, tuning and testing is included) and validation data
    val_split_start_date = '3/09/2023 00:00:00' #'1/08/2023 00:00:00'  #'3/09/2023 00:00:00'
    val_split_end_date = '20/09/2023 00:00:00' #'1/01/2025 00:00:00' #'20/09/2023 00:00:00'
    
    training_data = data[(data.index >= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S')) & (data.index < pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S'))]
    validation_data = data[(data.index >= pd.to_datetime(val_split_start_date, format='%d/%m/%Y %H:%M:%S')) & (data.index <= pd.to_datetime(val_split_end_date, format='%d/%m/%Y %H:%M:%S'))]
    
    # Scale data
    scaler_X = None # StandardScaler() # RobustScaler() StandardScaler()  MinMaxScaler()
    scaler_Y = None
    
    if scaler_X is not None:
        X = training_data.drop(columns=[target_name])  # Exclude the target column
        scaler_X.fit(X)
        X_scaled = scaler_X.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=training_data.index, columns=X.columns)
        
        Y = training_data[target_name]
        Y_scaled = Y
        #scaler_Y.fit(Y.values.reshape(-1, 1))
        #Y_scaled =  scaler_Y.transform(Y.values.reshape(-1, 1))
        training_data.loc[:, target_name] = Y_scaled
    
  
    if train_new_model: # train new model 
        # Train ML model 
        train_data = copy.deepcopy(training_data) # make copy of data to train model on
        
        # Correlation analysis
        correlate = Correlate(data=train_data, target_col_name=target_name)
        correlate.correlate_rates(feature_names=train_data.columns)
        correlate.importance(param_dict=param_dict, top_n=30)
        correlate.importance_plot()
        #train_data = correlate.filtered # option to retain only the most important features
        plt.show()
        
        training = Learn(train_data=train_data, target_col_name=target_name, weight_method='time*density', c_log=500, a_weight=0.5, scaler_X=scaler_X, scaler_Y=scaler_Y) # best to hypertune parameters for an average weight method value then play around with values
        training.train(param_dict=param_dict, hyper_tune=False)
        training.pickle_model()
        plt.show()
        
    if validate_model:  
        ## Validate model ##
        validation = Validate(test_data=validation_data, model_name=target_name, target_col_name=target_name, scaler_X=scaler_X, scaler_Y=scaler_Y)
        validation.validate_previous()   
    
    return data

def make_predictions(feature_values, model_name):
    parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive
    trained_model = pickle.load(open(parent_folder / 'models' / '{} Model.pkl'.format(model_name) ,'rb'))
    prediction = trained_model.predict(pd.DataFrame(data=feature_values, index=[0]))
    return prediction

if __name__ == "__main__":
    # Analyse plant data & group by CPM Rate
    train_EV1() 
    train_EV2()
    train_RB1()
    train_RB2()
    train_turbine(process_raw_data = True, filter_data=True, train_new_model = True, validate_model=True)
    train_natural_gas(process_raw_data = True, filter_data=True, train_new_model = True, validate_model=False)
    train_PB2(process_raw_data = True, filter_data=True, train_new_model=True, validate_model=True)

    # Example prediction for NG model at a single state
    feature_values = {'WW Moisture %':60, # WW moisture content %
                            'Air Temp':15, # air temperature
                            'WW Screw Ratio':1.1, #  WW screw feeder upper ratio
                            'Vent FLR P':1.9, # vent flowrate on LP header
                            'PB2 Steam Generation':97.5, # steam demand from PB2 t/hr
                            'Shift Letter': LabelEncoder.fit_transform(['A'])[0], # shift letter,
                            'LP_EVP_sum':133.9, 
                            'PM_sum':0, 'PD_sum':33.9, 'BPM_sum':0, 'CPM_sum':56.0, 'BLP_sum':16.4, 'HPM_sum':0, 'CST_sum':6.9, 'CEP_sum':2.5,
                            'Total Steam Demand':249.5,
                            'CPM On/Off':1, 'PD On/Off':1, 'BPM On/Off':0, 'PM On/Off':0, 'HPM On/Off':0, # binary denoting plant on/off
                            'RB1 TDS/d':884, 'RB2 TDS/d':839, # tonnes dry solids per day in RB TDS/d
                            'RB Steam Prod':219.5, # RB steam production t/hr
                            'RB1 TDS/d--1':884, 'RB2 TDS/d--1':839, # previous 5 mins total dry solids per day in RB
                            'Rainfall Sum':0 # total rain in past 24 hours mm
                            }
   
    make_predictions(feature_values, 'Total NG')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              