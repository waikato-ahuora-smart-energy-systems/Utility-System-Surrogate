'''
__author__ = 'Keegan Hall'
__credits__ = 

Class for performing correlation analysis for machine learning applications
'''
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
#import dcor

## just show correlations not necessarily filter by them
class Correlate():
    def __init__(
            self,
            data, # plant data for training
            target_col_name, # name of target column header
            ): 
        
        self.data = data
        self.target_col_name = target_col_name
        self.filtered = pd.DataFrame() # Create an empty DataFrame to store the filtered data
    
    def correlation(
            self): 
        '''
        Calculate variable correlation 

        Uses pandas correlation function to determine linear correlation 
        ''' 

        self.feature_matrix = self.data.corr() 

    def importance(
            self,
            param_dict = {}, # dictionary of training parameters 
            top_n=15, # number of top features to display
            ): 
        '''
        Calculate feature importance 

        Performs feature engineering (interaction and/or polynomial) and uses XGBoost to determine feature importance
        ''' 
        #self.data = self.data[:1000]
        # Convert index to datetime
        self.data.index = pd.to_datetime(self.data.index)
        # Split target and features
        X = self.data.drop(self.target_col_name, axis=1)  # Features
        self.target_data = self.data[self.target_col_name]  # Target variable
        
        feature_engineering = None # 'polynomial' # None #'interaction'
        if feature_engineering == 'interaction':
            # Create interaction features
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(X.columns) # column names of each feature including interactions
        
        elif feature_engineering == 'polynomial':
            #   Create polynomial features
            degree = 2
            feature_names = X.columns.tolist()
            X_poly = X.copy()
            for column in X.columns:
                for power in range(2, degree + 1):
                    new_column_name = f'{column}^{power}'
                    X_poly[new_column_name] = X[column] ** power
                    feature_names.append(new_column_name)

        elif feature_engineering == None:
            feature_names = X.columns.tolist()
            X_poly = X.values.tolist()
        
        # Define model
        self.expanded_data = pd.DataFrame(X_poly, columns=feature_names, index=self.data.index)   # create dataframe that includes existing and new features (if performed)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, self.target_data, test_size=0.2, random_state=42)
        xgb_model = xgb.XGBRegressor(**param_dict)
        
        
        # Train model
        self.trained_model = xgb_model.fit(X_poly, self.target_data)
        
        '''
        # Calculate model performance 
        y_pred = self.trained_model.predict(X_test) # test the model on test data
        res_2_score = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()}) # create dataframe of actual and predicted values 
        
        # Print metrics
        print(comparison_df)
        print(f"R2 score: {res_2_score}", )
        print(f"Mean Squared Error: {mse:.2f}")
        '''

        # Get feature importances
        importances = self.trained_model.feature_importances_

        # Create a DataFrame for feature importances
        self.feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        
        # Extract top N features
        self.top_n = top_n  # For example, select the top 10 features
        self.top_features = self.feature_importances.nlargest(self.top_n, 'importance')
        self.top_features_names = self.top_features['feature'].tolist()
        
        # Display top feature names
        # print("Top features:")
        # print(self.top_features)

        # print("All features:")
        # print(self.feature_importances)
     
    def distance_correlation(self):
        correlations = {}
        for col1 in self.data.columns:
            dcor_value = dcor.distance_correlation(self.data[col1], self.data[self.target_col_name])
            correlations[col1] = dcor_value
            
            
            # for col2 in self.data.columns:
            #     matrix.loc[col1, col2] = dcor.distance_correlation(self.data[col1], self.data[col2])
        
        # Convert to DataFrame for better visualization
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Distance Correlation'])
        print(correlation_df)
        
        # Visualize distance correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_df, annot=True, cmap="coolwarm")
        plt.title("Distance Correlation Matrix")
        plt.show()
    
         
    def cor_plot(
            self):
        '''
        Plot correlation matrix

        Uses pandas correlation function to determine linear correlation 
        ''' 
        # Setup colour map
        cmap = sns.diverging_palette(230, 20, as_cmap=True) # se
        N = self.data.shape[1]
        E = np.eye(N)

        # Mask upper triangle since data is redundant/repeated
        mask = np.triu(np.ones_like(self.feature_matrix, dtype=bool)) 
        mask[np.triu_indices_from(mask)] = True


        # Setup correlation heatmap
        fig, ax = plt.subplots(figsize=(10,6))         # figsize in inches
        sns.heatmap(
            self.feature_matrix,
            cmap=cmap,
            mask=mask,
            annot=True,
            annot_kws={'size': 10},
            vmin=-1, vmax=1,
            fmt=".2f",
            xticklabels=True, 
            yticklabels=True,
            ax=ax)

        # Adjust plot settings
        #fig = plt.figure(figsize=(10, 6))
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid")
        ax.tick_params(left=True, bottom=True) # add ticklines to axis
        ax.figure.subplots_adjust(left = 0.18, bottom = 0.28) # adjust border spacing
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light', fontsize='small') # adjust x labels
        ax.set_yticklabels(ax.get_yticklabels(), fontweight='light', fontsize='small') # adjust y labels
        cbar = ax.collections[0].colorbar # Modify the colorbar's decimal points
        cbar.set_ticks([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]) # Add a colorbar and modify its decimal points
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # Display 1 decimal points
        
        return fig


    def importance_plot(
            self):
        '''
        Plot feature importance 

        Plots bar graph of feature importance 
        ''' 
        
        # Plot the top N feature importances
        fig, ax = plt.subplots(figsize=(12,10))         # figsize in inches
        plt.bar(x=self.top_features['feature'], height=self.top_features['importance'], color='skyblue')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.title(f'Top {self.top_n} Feature Importances', fontsize=14)
        plt.tight_layout()

        fig2, ax1 = plt.subplots(figsize=(12,10))         # figsize in inches
        plt.bar(x=self.feature_importances['feature'], height=self.feature_importances['importance'], color='skyblue')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.title('All Feature Importances', fontsize=14)
        plt.tight_layout()
        plt.show()


        self.feature_matrix = self.feature_importances

    def cor_filter(
            self,
            cor_min_threshold = -0.2,
            cor_max_threshold = 0.2,
            keep_cols = []): # list of strings containing col names to keep reguardless of threshold 
        '''
        Filter data from correlation value

        Filter data columns that have a low correlation to a target column 
        ''' 

        # Iterate through columns and apply the threshold
        for column in self.data.columns: # column name
            correlation_value = self.feature_matrix[self.target_col_name][column]
            if correlation_value >= cor_max_threshold or correlation_value <= cor_min_threshold or column in keep_cols: # only save columns to self.filtered if they meet the threshold or are part of keep group
                self.filtered[column] = self.data[column]


    def importance_filter(
            self,
            keep_cols = [], # list of strings containing col names to keep reguardless of threshold 
            ): 
        '''
        Filter data from feature importance value

        Filter data columns that have a low feature importance
        ''' 
        # Add back user specified columns if they didnt meat the importance threshold
        feature_names = self.top_features['feature'].tolist() # start with top features ## check if top_features was modified
        for col in keep_cols: 
            if col not in self.top_features['feature'].values: # check if the column was in the top features
                feature_names.append(col)
                
            
        # Display top feature names
        # print("Features to keep:")
        # print(feature_names)

        # Filter/slice the original DataFrame based on the top N features
        self.filtered = self.expanded_data[feature_names].copy()

        # Add back target data
        self.filtered[self.target_col_name] = self.target_data.tolist()
    
        # Display the result
        print(self.filtered)

    def correlate_rates(self, feature_names):
        '''Correlate the target data with the specified features
        
        Use cross correlation to determine the lag between the target data and the specified features
        '''
      
        # Extract the relevant columns for cross-correlation
        target_data = self.data[self.target_col_name]
        
        for name in feature_names:
            feature_i_data = self.data[name]

            # Recalculating cross-correlation since the variables are not defined
            lags_new = np.arange(-len(feature_i_data) + 1, len(feature_i_data))
            cross_corr_new = np.correlate(
                feature_i_data - np.mean(feature_i_data), 
                target_data - np.mean(target_data), 
                mode='full'

            )

            cross_corr_new /= (np.std(feature_i_data) * np.std(target_data) * len(feature_i_data))

            # Find the lag with the highest absolute correlation
            max_corr_index = np.argmax(np.abs(cross_corr_new))
            max_corr_lag = lags_new[max_corr_index]
            max_corr_value = cross_corr_new[max_corr_index]

            # Plot the cross-correlation with the maximum highlighted
            #plt.figure(figsize=(10, 6))
            plt.plot(lags_new, cross_corr_new)
            plt.axhline(0, color='black', linestyle='--')
            plt.scatter([max_corr_lag], [max_corr_value], color='red', label=f"Max Corr: {max_corr_value:.2f} at Lag {max_corr_lag}")
            plt.xlabel("Time Step Lag")
            plt.ylabel("Correlation Coefficient")
            #plt.grid()
            plt.legend()
            plt.show()



            # Output the lag and correlation value

            max_corr_lag, max_corr_value    

    def save_features(
            self,
            filename): # filename/identifier of data
        '''Save correlation matrix & data'''
        
        # Remove the file format from the raw data file name
        sub_list = [".csv", ".xlsx"]
        for sub in sub_list: 
            name = filename.replace(sub , '') 

        parent_folder = Path(__file__).parent.parent # location of the folder containing this file relative to your C drive
        
        # Save correlation matrix
        file_to_save = parent_folder / "outputs" / "matrix" / "{} Feature Matrix.xlsx".format(name)  # CWD relative file path for input files
        self.feature_matrix.to_excel(file_to_save, index=True) # save matrix to excel

        # Save the filtered data
        file_to_save = parent_folder / "outputs" / "filtered" / "Feature Filtered {}.csv".format(name)  # CWD relative file path for input files
        self.filtered.to_csv(file_to_save, index=True)
        





        