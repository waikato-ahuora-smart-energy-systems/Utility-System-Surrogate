'''
__author__ = 'Keegan Hall'
__credits__ = 

Class for machine learning of industrial plant data
'''
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pickle
import calendar
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors, KernelDensity
from scipy.stats import gaussian_kde, chisquare, kstest
from denseweight import DenseWeight
from sklearn.mixture import GaussianMixture
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



from hyperopt import STATUS_OK, Trials, fmin, hp, tpe




parent_folder = Path(__file__).parent.parent # location of the folder containing this file relative to your C drive

class Learn():
    def __init__(
        self,
        train_data, # plant data
        target_col_name, # name of target column header
        c_log = 650, # constant to shift the logarithm (ensures log(0) is not undefined)
        a_weight = 0.1, # alpha weight for the dense weight
        weight_method = 'time', # method to use for weighting data
        scaler_X='', scaler_Y='', # scalers for X and Y data
        ): 
        '''Class init'''
        
        self.train_data = train_data
        self.target_col_name = target_col_name
        self.c_log = c_log
        self.a_weight = a_weight
        self.weight_method = weight_method
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        
        
    def train(
        self,
        param_dict = {}, # dictionary of training parameters 
        hyper_tune = False, # whether to hyper tune the model
    ): 
        '''
        Train model 

        Splits target data an trains a NN model with custom energy balance
        ''' 
        #self.train_data = self.train_data[:5]
        
        ## Train & test model ##
        X_train, y_train, X_test, y_test, X_val, y_val, model = self.setup_model(param_dict)
        weights = self.weight_data(y_train)
        self.fit_model(hyper_tune=hyper_tune, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test, model=model, weights=weights)
        self.test_model(X_test=X_test, y_test=y_test, )
        
        
        ## Retrain model on entire dataset ##
        X = pd.concat([X_train, X_test, X_val]) 
        Y = pd.concat([y_train, y_test, y_val])
        self.trained_model = model.fit(X, Y, sample_weight=self.weight_data(Y), verbose=False)  # Fit model 
    
    def setup_model(self, 
                    param_dict = {}, # model training parameters
                    ):
        # Convert index to datetime
        self.train_data.index = pd.to_datetime(self.train_data.index)
        
        
        # Split target and features
        X = self.train_data.drop([self.target_col_name], axis=1)  # Features
        Y = self.train_data[self.target_col_name]  # Target variable
        
        # Weekly split data v2
        
        # Get week-year pairs (use year + week to avoid cross-year clashes)
        calendar = X.index.isocalendar()
        week_year = list(zip(calendar.year, calendar.week))

        # Keep only unique week-year pairs, sorted chronologically
        unique_weeks = sorted(set(week_year))

        # Prepare empty lists for indices
        train_indices = []
        val_indices = []
        test_indices = []

        # Slide over 10-week blocks
        block_size = 10
        for i in range(0, len(unique_weeks), block_size):
            block = unique_weeks[i:i+block_size]
            if len(block) < 10:
                continue  # skip incomplete final block (or handle specially if you want)

            train_weeks = block[:6]
            val_weeks = block[6:8]
            test_weeks = block[8:10]

            # Match indices based on these week-year pairs
            for df_index in X.index:
                y, w = df_index.isocalendar().year, df_index.isocalendar().week
                if (y, w) in train_weeks:
                    train_indices.append(df_index)
                elif (y, w) in val_weeks:
                    val_indices.append(df_index)
                elif (y, w) in test_weeks:
                    test_indices.append(df_index)
      
        # Final splits
        X_train = X.loc[train_indices]
        X_val = X.loc[val_indices]
        X_test = X.loc[test_indices]

        y_train = Y.loc[train_indices]
        y_val = Y.loc[val_indices]
        y_test = Y.loc[test_indices]
        
        print('shapes', X_train.shape, X_val.shape, X_test.shape)
        
        def plot_split(y_train, y_val, y_test):
            import datetime
            from matplotlib.collections import LineCollection
            # Combine data
            y_all = pd.concat([y_train, y_val, y_test])
            y_all = y_all.sort_index()  # Ensure chronological order
            x = mdates.date2num(y_all.index)
            y = y_all.values 

            # Create segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Assign colors based on which dataset the segment belongs to
            colors = []
            for i in range(len(y_all) - 1):
                current_index = y_all.index[i]
                if current_index in y_train.index:
                    colors.append('blue')
                elif current_index in y_val.index:
                    colors.append('red')
                else:
                    colors.append('green')

            # Create line collection
            #idx = np.random.choice(len(segments), size=len(segments)//2, replace=False)
            lc = LineCollection(segments, colors=colors, linewidths=1.0)
            fig, ax = plt.subplots()
            ax.add_collection(lc)

            # Axes formatting
            ax.set_xlim(mdates.date2num(datetime.date(2022, 7, 1)), mdates.date2num(datetime.date(2022, 9, 1)))
            ax.set_ylim(y.min(), y.max())
            ax.set_ylabel('Target Value')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

            # Legend proxy
            from matplotlib.lines import Line2D
            legend_lines = [
                Line2D([0], [0], color='blue', label='Training'),
                Line2D([0], [0], color='red', label='Tuning'),
                Line2D([0], [0], color='green', label='Testing')
            ]
            ax.legend(handles=legend_lines, loc='upper center')

            plt.show()

        ## Create ML model type ##
        self.model_type = 'xgboost'
        if self.model_type == 'xgboost':
            model = xgb.XGBRegressor(**param_dict)
            
        elif self.model_type == 'NN':
            # Scale the data for NN
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test = pd.DataFrame(scaler.fit_transform(X_test), index=X_test.index, columns=X_test.columns)
            
            self.scaler2 = StandardScaler()
            #y_train = self.scaler2.fit_transform(y_train.values.reshape(-1, 1)) # pd.DataFrame(scaler2.fit_transform(y_train.values.reshape(-1, 1)), index=X_train.index, columns=self.target_col_name) 
            #y_train = pd.DataFrame(y_train, index=X_train.index, columns=[self.target_col_name])
            
            # Define custom loss function
            def custom_loss(y_true, y_pred):
                
                # Convert penalties to a tensor
                #penalties_tensor = tf.convert_to_tensor(energy_balance_train, dtype=tf.float32)
                #y_tensor = tf.convert_to_tensor(pd.DataFrame(data={'target': y_train.values}).values, dtype=tf.float32)  # Full dataset targets as tensor where the data has been reshaped
                
                # Compare each row in y_true with y to find matching rows
                #matches = tf.equal(tf.expand_dims(y_true, axis=1), tf.expand_dims(y_tensor, axis=0))  # Expand dims for broadcasting
                #row_indices = tf.where(tf.reduce_all(matches, axis=-1))[:, 1]  # Get matching indices
            
                # Gather the penalties corresponding to the indices of y_true in the batch
                #batch_penalties = tf.gather(penalties_tensor, row_indices)
                
                # Compute the base loss (Mean Squared Error for batch)
                mse = tf.square(y_true - y_pred)
                mae = tf.abs(y_true - y_pred)
                
                # Combine with the penalty
                total_loss = mae 
                '''
                tf.print("Matches:", matches)
                tf.print("Row Indices:", row_indices)
                tf.print('true', y_true,)
                tf.print('pred', y_pred,)
                tf.print('penalty', batch_penalties)
                tf.print('total loss',total_loss)
                tf.print('##########')
                '''
                return total_loss

            # Build a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(1)  # Single output for regression
            ])
            
            # Compile the model with the custom loss function
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Set the learning rate to 0.001
            model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
   
        return X_train, y_train, X_test, y_test, X_val, y_val, model
                
    def weight_data(self, target_data):   
         ## Assign weights to data based upon target density ##
        
        if self.weight_method == 'density':
            # Calculate rarity of data using denseweight
            dw = DenseWeight(alpha=self.a_weight)
            weights = dw.fit(target_data.values)
            plt.scatter(target_data, weights, s=1)
            plt.show()  
                
        elif self.weight_method == 'BinFreq':
            # Create bins
            bins = np.linspace(target_data.min(), target_data.max(), num=50)  # Adjust 'num' for bin count
            bin_indices = np.digitize(target_data, bins)

            # Calculate frequency per bin
            bin_counts = np.bincount(bin_indices, minlength=len(bins))
            weights = 1 / (bin_counts[bin_indices])  # Add small value to avoid division by zero
        elif self.weight_method == 'kNN':
            # Example target data
            target_data = target_data.values.reshape(-1, 1)  # Reshape for NearestNeighbors

            # Fit nearest neighbors
            knn = NearestNeighbors(n_neighbors=10)  # Adjust 'n_neighbors' for sensitivity
            knn.fit(target_data)
            distances, _ = knn.kneighbors(target_data)

            # Rarity is the inverse of average distance to neighbors
            weights = (distances.mean(axis=1) + 1e-6)

        elif self.weight_method == 'GMM':
            gmm = GaussianMixture(n_components=3, random_state=42)  # Adjust 'n_components' based on target distribution
            gmm.fit(target_data.values.reshape(-1, 1))
            probabilities = gmm.score_samples(target_data.values.reshape(-1, 1))

            # Rarity is the inverse of probability density
            weights = 1 / (np.exp(probabilities) + 1e-6)

            # Normalize rarity
            weights /= weights.sum()
        elif self.weight_method == 'manual':   
            weights = np.ones(len(target_data))
            for i in range(len(target_data)):
                if target_data.iloc[i] > 1500 and target_data.iloc[i] < 2000:
                    weights[i] = 0.8
                elif target_data.iloc[i] > 2000 and target_data.iloc[i] < 2500:
                    weights[i] = 0.6
                else:
                    weights[i] = 1
        elif self.weight_method == 'time*density':
            time_index = np.arange(len(target_data))  # Replace `y` with your target data
            
            # Logarithmic growth of weights over time
            weights_log = np.log(time_index + self.c_log) / np.log(10)
            
            dw = DenseWeight(alpha=self.a_weight)
            density = dw.fit(target_data.values)
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Constantia"],
                "font.size": 12,
                "figure.figsize": (8, 6)
                #"figure.dpi": 500
                })  # Set the figure size to A4 dimensions and DPI to 1000

            weights = weights_log  * density
            fig, ax = plt.subplots()
            idx = np.random.choice(len(weights), size=len(weights)//2, replace=False)
            ax.scatter(target_data*42/1000, weights, s=1)
            ax.set_xlabel("Total Gas Consumption (GJ/h)")
            ax.set_ylabel("Sample Weight")
            ax.legend()
            #fig.savefig(parent_folder / "figure data" / 'Gas sample weights.png', dpi=300, bbox_inches="tight")
            plt.show()
            
        else:
            weights = np.ones(len(target_data))
        
        return weights
        
    def fit_model(self, 
                  hyper_tune = False, # True/False logic to hyper tune the model
                  X_train = pd.DataFrame, y_train = pd.DataFrame, # training data set
                  X_val = pd.DataFrame, y_val = pd.DataFrame, # validation data set
                  model = '',
                  weights = [],
                  ): # base ML model
     
        
        ##  Hyperparameter tuning ##
        if hyper_tune: # define search space parameters   
            search_space = {
                'n_estimators': hp.quniform('n_estimators', 100, 800, 50),    # Range: 300–1000, step: 50
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),        # Range: 0.01–0.3
                'max_depth': hp.quniform('max_depth', 4, 12, 2),                # Range: 4–12
                'subsample': hp.quniform('subsample', 0.5, 1.0, 0.1),       # Range: 0.6–1.0
                'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),# Range: 0.6–1.0
                'lambda': hp.uniform('lambda', 0.0, 5.0),                         # L2 Regularization
                # option to tune weigting parameters
                # 'c_log': hp.quniform('c_log', 50, 1000, 50),   
                # 'a_weight': hp.uniform('a_weight', 0.0, 2.0),             
            
            }
            
            # Set up hyperopt objective
            def objective(params):
                
                # Train XGBoost model with given parameters
                sub_model = xgb.XGBRegressor(
                            n_estimators=int(params['n_estimators']),
                            learning_rate=params['learning_rate'],
                            max_depth=int(params['max_depth']),
                            subsample=params['subsample'],
                            colsample_bytree=params['colsample_bytree'],
                            reg_lambda=params['lambda'],
                            random_state=42
                        )
                '''
                # Set up weights - optional if included in search_space
                c_log=params['c_log']
                a_weight=params['a_weight']
                
                time_index = np.arange(len(y_train))  # Replace `y` with your target data
                weights_log = np.log(time_index + c_log) / np.log(10)
                dw = DenseWeight(alpha=a_weight)
                density = dw.fit(y_train.values)
                weights = weights_log  * density
                '''
                
                # Train the model on sequential split
                sub_model.fit(X_train, y_train, sample_weight=weights, verbose=False)
                
                # Predict and evaluate
                y_pred = sub_model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
                res2 = r2_score(y_val, y_pred)
                mape = np.mean(np.abs((y_val - y_pred)/y_val))*100
               
                return {'loss': mse, 'status': STATUS_OK}

            
            # Run hyperopt optimization    
            trials = Trials()  # To store results of each iteration
            best_params = fmin(
                fn=objective,  # Objective function
                space=search_space,  # Hyperparameter space
                algo=tpe.suggest,  # Tree-structured Parzen Estimator
                max_evals=500,  # Number of iterations
                trials=trials,  # To keep track of results
            )
            
            # Extract best parameters
            final_params = {
                    'n_estimators': int(best_params['n_estimators']),
                    'learning_rate': best_params['learning_rate'],
                    'max_depth': int(best_params['max_depth']),
                    'subsample': best_params['subsample'],
                    'colsample_bytree': best_params['colsample_bytree'],
                    'lambda': best_params['lambda'],
                    # 'c_log': best_params['c_log'],
                    # 'a_weight': best_params['a_weight'],
                }
            
            print("Final parameters:", final_params)
           
            # Train model
            model = xgb.XGBRegressor(**final_params)
            self.trained_model = model.fit(X_train, y_train, sample_weight=weights, verbose=False)  # Fit model on entire dataset
           
        else: # train with user specified params
            pass
           
            '''
            # Automatically undersample the data
            import ImbalancedLearningRegression as iblr


            # Merge X_train and y_train into a DataFrame
            data = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
            train_data_rbl =  iblr.random_under(data = data, y = self.target_col_name ) 
        
            y_train = train_data_rbl[self.target_col_name]
            X_train = train_data_rbl.drop(self.target_col_name, axis=1)
                        
            sns.kdeplot(y_train, label="Original", color='blue')
            sns.kdeplot(train_data_rbl[self.target_col_name], label="Undersampled", color='red')
            plt.legend()
            plt.show()
            '''
            
            
            '''
            # Manually undersample the data
            lower_bound = 1100
            upper_bound = 4000
            data_to_remove = 70/100
            mask = (y_train >= lower_bound) & (y_train <= upper_bound)
            indices_to_remove = np.random.choice(np.where(mask)[0], size=int(np.sum(mask) * data_to_remove), replace=False)
            X_train = np.delete(X_train, indices_to_remove, axis=0)
            y_train = np.delete(y_train, indices_to_remove, axis=0)
            weights = np.delete(weights, indices_to_remove, axis=0)
            
            
            lower_bound = 120
            upper_bound = 140
            data_to_remove = 70/100
            mask = (y_train >= lower_bound) & (y_train <= upper_bound)
            indices_to_remove = np.random.choice(np.where(mask)[0], size=int(np.sum(mask) * data_to_remove), replace=False)
            X_train = np.delete(X_train, indices_to_remove, axis=0)
            y_train = np.delete(y_train, indices_to_remove, axis=0)
            weights = np.delete(weights, indices_to_remove, axis=0)
            '''    
            
            
        
        if self.model_type == 'xgboost': # train Xgboost model
            self.trained_model = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=weights, verbose=False)  # Fit model 
        elif self.model_type == 'NN': # train NN model
            model.fit(X_train, y_train, epochs=50, batch_size=32, sample_weight=weights, verbose=1)    
            self.trained_model = model
        
        
    def test_model(self, 
                   X_test = pd.DataFrame, 
                   y_test= pd.DataFrame,
                   ):
        unit = "GJ/h"
        target_label = f'Total Gas Consumption ({unit})'
        
        ## Calculate model performance ##
        y_pred =  pd.Series(self.trained_model.predict(X_test).flatten(), index=y_test.index) # test the model on test data
        print(X_test.info())
        y_pred = y_pred * 42 / 1000 # convert to GJ/h
        y_test = y_test * 42 / 1000 # convert to GJ/h
        if self.scaler_Y is not None:
            y_pred_list = self.scaler_Y.inverse_transform(y_pred.to_numpy().reshape(-1, 1)).flatten() 
            y_test_list = self.scaler_Y.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()
            
            y_pred = pd.Series(y_pred_list, index=y_test.index)    
            y_test = pd.Series(y_test_list, index=y_test.index)
            
    
        res_2_score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
        residuals = (y_test- y_pred)
     
        
        self.comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residuals':residuals}) # create dataframe of actual and predicted values 

        ## Print metrics ##
        print(self.comparison_df)
        print(f"R2 score: {res_2_score}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}")
        
        # Order residuals by worst at the top and show corresponding time
        sorted_residuals = self.comparison_df.sort_values(by='Residuals', ascending=False)  
        print(sorted_residuals.head(20))
        self.train_data.info()    
        
        ## Plot residuals identified by month ##
        unique_months = np.unique(X_test.index.month)
        fig, ax = plt.subplots(figsize=(3.1, 6))  # Create a figure with specified size and DPI

        # Scatter plot per month
        for month in unique_months:
            month_name = calendar.month_name[month]
            mask = X_test.index.month == month
            ax.scatter(
                y_pred[mask],
                y_test[mask],
                label=f"{month_name}",
                s=2,
                alpha=0.7
            )

        # Perfect fit line
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k', linewidth=2, label="Perfect Fit")

        # Calculate axis limits
        max_val = max(max(y_test), max(y_pred)) * 1.05  # add 5% buffer for visibility
        x_vals = np.linspace(0, max_val, 100)

       
        # Labels and limits
        ax.set_ylabel(f"Actual {target_label}")
        ax.set_xlabel(f"Predicted {target_label}")
        ax.set_ylim([0, max_val])
        ax.set_xlim([0, max_val])
        ax.legend(  loc='upper center', 
                    bbox_to_anchor=(0.5, -0.12),  # Position below the plot
                    ncol=3, 
                    columnspacing=0.55,)                     # Number of columns to spread items acros fontsize='small')
        plt.show()
        fig.savefig(parent_folder / "figure data" / 'EVP1.png', dpi=300, bbox_inches="tight")
       

        ## Plot feature importance ##
        xgb.plot_importance(self.trained_model, importance_type='gain')
       
        ## Plot residuals & target values ##
        fig = plt.figure()
        ax1 = fig.gca()  # residual
        ax2 = ax1.twinx()  # target value
        ax1.plot(X_test.index, y_pred, label=f"Predicted {target_label}", color='blue')
        ax1.plot(X_test.index, y_test, label=f"Actual {target_label}", color='red')
        ax1.set_ylabel("Total Gas Consumption (GJ/h)")
        fig.legend()
        
    
        ## Plot metrics for different bands of the target value ##
        bands = range(int(self.comparison_df['Actual'].min()), int(self.comparison_df['Actual'].max()) + 10, int(2))
        band_rsme = []
        for i in range(len(bands)-1):
            lower_bound = bands[i]
            upper_bound = bands[i+1]
            values_within_band = self.comparison_df[(self.comparison_df['Actual'] >= lower_bound) & (self.comparison_df['Actual'] < upper_bound)].index.tolist()
            
            # Create dataframe with actual and predicted values
            df_predicted_values = pd.DataFrame({'Actual': self.comparison_df.loc[values_within_band, 'Actual'], 'Predicted': self.comparison_df.loc[values_within_band, 'Predicted']})
            #print(df_predicted_values)
            actual_values = df_predicted_values['Actual'] 
            predicted_values = df_predicted_values['Predicted'] 
            try:
                rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
                mape = np.mean(np.abs((actual_values - predicted_values)/actual_values))*100
            except:
                rmse = np.nan
                mape = np.nan
            band_rsme.append(rmse)
            print(f"{lower_bound}- {upper_bound} NM3/hr RMSE: {rmse:.2f}, MAPE: {mape:.2f}, Percentage of Data: {(len(values_within_band) / len(self.comparison_df)) * 100:.2f}%")
         
        fig = plt.figure()
        ax1 = fig.gca() # model rmse 
        ax2 = ax1.twinx() # data density 
        ax1.plot(bands[:-1], band_rsme, label='RMSE', marker='o')  
        sns.kdeplot(y_test, label="Data Density", color='red', ax=ax2)  
       
        ax1.set_xlabel(f"Actual {target_label}")
        ax1.set_ylabel(f"RMSE ({unit})")  
        ax2.set_ylabel("Data Density")
        ax1.set_ylim([0, max(band_rsme)])  # Set y-axis range from 0 to the maximum value
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Set ax2 to scientific numbering
        fig.legend(loc='upper center', bbox_to_anchor=(0.55, 0.9)) # ValueError: 'center top' is not a valid value for loc; supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        plt.show()
      
    def pickle_model(
        self,
        ):
        '''
        Pickle ML model

        Save model to pickle so that it can be used later on
        '''
        file_to_save = parent_folder / 'models' / '{} Model.pkl'.format(self.target_col_name)  
        with open(file_to_save, 'wb') as model_file:
                    pickle.dump(self.trained_model, model_file)

    def check_residuals(
        self,
        alter_groups,
        ):
        '''
        Check the residuals for each feature 

        Calculates prediction residuals and plots them against each feature for user checking
        ''' 
        # Open pickle
        file_to_open = parent_folder / 'outputs' / 'models' / '{} model.pkl'.format(self.target_col_name)  
        self.trained_model = pickle.load(open(file_to_open,'rb'))
        
        # Split target and features
        X = self.train_data.drop(self.target_col_name, axis=1)  # Features
        Y = self.train_data[self.target_col_name]  # Target variable
        
        # Calculate model performance 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        y_pred = self.trained_model.predict(X_test) # test the model on test data
        residuals = (y_test - y_pred.flatten()).tolist()
        res_2_score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"R2 score: {res_2_score}", )
        print(f"Mean Squared Error: {mse:.2f}")

        # Plot residuals
        
        for group in alter_groups:  # iterate across each steam group
            plt.scatter(X_test[group], residuals, s=1)
            plt.xlabel(f'{group} (t/hr)')
            plt.ylabel('NG Residual (NM3/hr')
            plt.figure()
     
        plt.show()
        

           
                    


        
    
        
    


        