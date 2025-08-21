'''
__author__ = 'Keegan Hall'
__credits__ = 

Class for validating trained machine learning models
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import time
import calendar
import seaborn as sns
from scipy.signal import savgol_filter

parent_folder = Path(__file__).parent.parent # location of the folder containing this file relative to your C drive

class Validate():
    def __init__(
        self,
        test_data, # plant data for testing, has already been processed
        model_name, # name of model pickle file
        target_col_name, # name of target column
        scaler_X='', scaler_Y='', # scalers for X and Y data
        ): 
        '''Class init'''
        
        self.test_data = test_data
        self.model_name = model_name
        self.target_col_name = target_col_name
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
    
    def load_pickle(
        self,
        ): 
        '''
        Open pickle ML model

        Open previously trained model from pickle 
        '''
        # Open pickle
        file_to_open = parent_folder / 'models' / '{} Model.pkl'.format(self.model_name)  
        self.trained_model = pickle.load(open(file_to_open,'rb'))

       
    def validate_previous(
            self,
            ):
        '''
        Validate model on large new data set from previous time range

        Load previously trained model and predict target on new data set
        '''
        unit = "MW"
        target_label = f'Turbine Generation ({unit})'
        
        ## Setup data
        # Load model pickle
        self.load_pickle()

        # Split target and features
        X_test = self.test_data.drop([self.target_col_name], axis=1)  # Features
        y_test = self.test_data[self.target_col_name]  # Target variable
        
        # Apply scaler to test data
        if self.scaler_X is not None:
            X_test = pd.DataFrame(self.scaler_X.transform(X_test), index=X_test.index, columns=X_test.columns)
            
        if self.scaler_Y is not None:
            Y_scaled = self.scaler_Y.transform(y_test.values.reshape(-1, 1)).flatten()
            y_test = pd.DataFrame(Y_scaled, index=y_test.index, columns=[self.target_col_name])

        ## Calculate model performance ##
        y_pred =  pd.Series(self.trained_model.predict(X_test).flatten(), index=y_test.index) # test the model on test data
        
        if self.scaler_Y is not None:
            y_pred_list = self.scaler_Y.inverse_transform(y_pred.to_numpy().reshape(-1, 1)).flatten() 
            y_test_list = self.scaler_Y.inverse_transform(y_test.to_numpy().reshape(-1, 1)).flatten()
            
            y_pred = pd.Series(y_pred_list, index=y_test.index)    
            y_test = pd.Series(y_test_list, index=y_test.index)
           
        #y_pred = y_pred * 42 / 1000 # convert to GJ/h
        #y_test = y_test * 42 / 1000 # convert to GJ/h   
            
        res_2_score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
        residuals = (y_test - y_pred).tolist()
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
        
        unique_months = np.unique(X_test.index.month)
        fig, ax = plt.subplots()

        for month in unique_months:
            month_name = calendar.month_name[month]
            mask = X_test.index.month == month # mask for the current month
            ax.scatter(
                y_pred[mask],
                y_test[mask],
                label=f"{month_name}",
                s=2,  # Increase size for better visibility if needed
                alpha=0.7  # Add transparency for overlapping points
            )

        ax.axline([0, 0], [1, 1], color='black', linewidth=3)  # Add diagonal reference line
        ax.set_ylabel(f"Actual {target_label}")
        ax.set_xlabel(f"Predicted {target_label}")
        ax.set_ylim([0, max(max(y_test), max(y_pred))])  # Set y-axis range from 0 to the maximum value
        ax.set_xlim([0, max(max(y_test), max(y_pred))])  # Set y-axis range from 0 to the maximum value
        ax.legend(title='Month', loc='best', fontsize='small')

         ## Plot residuals & target values ##
        fig = plt.figure()
        ax1 = fig.gca()  # residual
        ax2 = ax1.twinx()  # target value
        ax1.plot(X_test.index, y_pred, label=f"Predicted", color='blue')
        ax1.plot(X_test.index, y_test, label=f"Actual", color='red')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))  
        ax1.set_ylabel(target_label)
        ax1.legend(loc='upper right')
        
        ## Plot metrics for different bands of the target value ##
        bands = range(int(self.comparison_df['Actual'].min()), int(self.comparison_df['Actual'].max()) + 10, int(2))
        band_rsme = []
        for i in range(len(bands)-1):
            lower_bound = bands[i]
            upper_bound = bands[i+1]
            values_within_band = self.comparison_df[(self.comparison_df['Actual'] >= lower_bound) & (self.comparison_df['Actual'] < upper_bound)].index.tolist()
            
            # Create dataframe with actual and predicted values
            df_predicted_values = pd.DataFrame({'Actual': self.comparison_df.loc[values_within_band, 'Actual'], 'Predicted': self.comparison_df.loc[values_within_band, 'Predicted']})
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
        fig.legend(loc='upper center', bbox_to_anchor=(0.55, 0.9)) 
    
    
    def check_residuals(
        self,
        ):
        '''
        Check the residuals for each feature 

        Calculates prediction residuals and plots them against each feature for user checking
        ''' 
        # Open pickle
        file_to_open = parent_folder / 'outputs' / 'models' / '{} Aug21-Jul22 Model.pkl'.format(self.target_col_name)  
        self.trained_model = pickle.load(open(file_to_open,'rb'))
        
        # Split target and features
        X = self.test_data.drop(self.target_col_name, axis=1)  # Features
        Y = self.test_data[self.target_col_name]  # Target variable
        
        # Calculate model performance 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.99, random_state=42)
        y_pred = self.trained_model.predict(X_test) # test the model on test data
        residuals = (y_test - y_pred.flatten()).tolist()
        res_2_score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"R2 score: {res_2_score}", )
        print(f"Mean Squared Error: {mse:.2f}")

        # Plot residuals
        for group in X_test.columns:  # iterate across each steam group
            plt.scatter(X_test[group], residuals, s=1)
            plt.xlabel(f'{group} (t/hr)')
            plt.ylabel('NG Residual (NM3/hr)')
            plt.figure()
        
        plt.show()

        


