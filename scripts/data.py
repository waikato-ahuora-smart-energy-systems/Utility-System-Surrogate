'''
__author__ = 'Keegan Hall'
__credits__ = 

Class for importing, processing and storing plant data for machine learning applications
'''
from pathlib import Path
from turtle import position, title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hampel import hampel
from collections import OrderedDict
from scipy.signal import savgol_filter
import csv

from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import CubicSpline
from sklearn.decomposition import KernelPCA

import seaborn as sns

from sklearn.cluster import k_means
from scipy.stats import gaussian_kde
from scipy.spatial.distance import mahalanobis, pdist, squareform
from numba import jit
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

from open_nipals.utils import ArrangeData
from sklearn.preprocessing import StandardScaler
from open_nipals.nipalsPCA import NipalsPCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


class Data():
    def __init__(
            self,
            parent_folder=None,
            target_col_name=''):
        if parent_folder is None:
            self.parent_folder = Path(__file__).parent.parent
        else:
            self.parent_folder = Path(parent_folder)

        self.raw = None
        self.filtered = None
        self.target_col_name = target_col_name

    def data_import(
            self, 
            file_to_open = '', 
            header_row = 2, # row of column description/header (starts from row 1)
            data_start_row = 18, # row of the actual data starting (starts from row 1)
            time_col_name = None, # name of time column header in data
            header_col = 'description', # switch between 'description' and 'tag' for header column
            ): 
        '''Import data from file '''
        
        # Adjust rows to 0-indexed for pandas
        header_row_index = header_row - 1
        data_start_row_index = data_start_row - 1
        skiprows_range = list(range(header_row_index + 1, data_start_row_index))
                         
        if file_to_open.suffix == '.csv':
            delimiter = ','  # Default delimiter is comma

            # Check if the file is delimited by a semicolon
            with open(file_to_open, 'r') as file:
                first_line = file.readline()
                if ';' in first_line:
                    delimiter = ';'

            df = pd.read_csv(file_to_open, 
                            header=None, 
                            delimiter=delimiter,
                            #nrows=1000,
                            )
            print('csv')
        elif file_to_open.suffix == '.xlsx':
            df = pd.read_excel(file_to_open, 
                                header=None, 
                                engine='openpyxl',
                                dtype={time_col_name: str}
                                )
            print('xlsx')
        else:
            raise ValueError("Invalid filetype, must be .csv or .xlsx")
        
        # Extract the header string, units string and numeric values
        try:
            units_row_idx = df[df.iloc[:, 0].str.contains("Units", na=False)].index[0]
            self.units = df.iloc[units_row_idx, 1:].astype(str)
        except:
            print('Units not provided in the file')
            self.units = np.nan
            
        tag_num = df.iloc[0]
        tag_name = df.iloc[header_row_index]
        self.raw = df.iloc[data_start_row_index:].copy()
        if header_col == 'description':
            self.raw.columns = tag_name
        elif header_col == 'number':
            self.raw.columns = tag_num
        self.raw = self.raw.replace('#DIV/0!', np.nan)# replace div/0 with NaN since excel import can through an error
        
           
        # Set time column as index
        descr_position = tag_name[tag_name == time_col_name].index[0] # find the position of the time column
        time_header_name = self.raw.columns[descr_position] # find the column name of the time column
        if time_col_name in tag_name.values:
            # Convert 'Descr' column to datetime index using specified format
            if self.raw[time_header_name].str.contains('\d{4}-\d{2}-\d{2}').any():
                self.raw[time_header_name] = pd.to_datetime(self.raw[time_header_name], dayfirst=False)
            else:
                self.raw[time_header_name] = pd.to_datetime(self.raw[time_header_name], dayfirst=True)
    
            self.raw = self.raw.set_index(time_header_name)
            
        else:
            pass
            #raise KeyError(f"Column {time_col_name} does not exist in the DataFrame.")
        
        # Function to convert values based on their type        
        def convert_values(value):
            if pd.isna(value) or value == "":
                return value  # Leave empty values as is (NaN or empty string)
            try:
                return float(value)  # Convert to float if possible
            except ValueError:
                return str(value)  # Otherwise, keep it as a string

        # Apply the conversion function to the entire DataFrame
        self.raw = self.raw.map(convert_values)
        print(self.raw)
        
      
    
    def process_units(self, scaling_units=(''), scale_factor=1):
        '''
        Processes initial data
        
        This function processes the unit of the data to ensue that the data is consistent
        '''
        import matplotlib.pyplot as plt
        # Scale numeric columns based on units
        for col in self.raw.columns:  
            if self.units.iloc[self.raw.columns.get_loc(col)] in scaling_units: # check if column has units that need to be scaled
                self.raw[col] = self.raw[col]/ scale_factor
        
    
    def apply_PCA(self, n_components = 20):
        '''Apply PCA to the data
        This function applies PCA to the data to reduce dimensionality and extract principal components
        However, it is not needed in the final model'''
        
        # Invoke preprocessing pipeline
        arrdat = ArrangeData()
        scaler = StandardScaler()
        PCA_data = self.filtered.copy().drop(columns=self.target_col_name)
        data = scaler.fit_transform(arrdat.fit_transform(PCA_data))
        
        # Apply PCA
        model = NipalsPCA(n_components=n_components)
        transformed_data = model.fit_transform(data)
        print(model.loadings)
        
        # Show variable loadings
        loadings_df = pd.DataFrame(model.loadings, columns=[f"PC{i+1}" for i in range(n_components)])
        loadings_df.insert(0, "Variable", PCA_data.columns)
        print(loadings_df)
        
        # Place principal components in a DataFrame
        self.pca_df = pd.DataFrame(transformed_data, index=self.filtered.index, columns=[f"PC{i+1}" for i in range(n_components)])
        self.pca_df[self.target_col_name] = self.filtered[self.target_col_name]
        

    def cluster_data(self, n_clusters=3, data=None, cluster_mask = 1):
        '''Cluster data using KMeans clustering
        This function clusters the data using KMeans clustering and plots the clusters over time
        '''
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        # Add to DataFrame
        data["cluster"] = cluster_labels
      
        # Plot the clusters over time
        self.clustered_df = data
        
        df = self.clustered_df.copy()
        df[self.target_col_name] = self.filtered[self.target_col_name] # add the target column back to the clustered data
        df = df.sort_index()    

        plt.figure(figsize=(14, 6))
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            plt.plot(
                cluster_data.index,
                cluster_data[self.target_col_name],
                label=f'Cluster {cluster_id}',
                linewidth=0.8,  # thinner lines for clarity
                alpha=0.7
            )

        plt.xlabel("Time")
        plt.ylabel("Target")
        plt.title("Target Over Time by Cluster")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Filter out cluster
        mask = data["cluster"].isin([cluster_mask])
        self.clustered_df = data[mask]
        self.clustered_df = self.clustered_df.drop(columns=["cluster"]) # drop the cluster column
        
        return mask
    
    def filter(
            self, 
            filter_cols): 
        '''
        Filters data

        Cleans data by removing missing values and periods of bad operations e.g shutdowns
        ''' 

        # Remove missing data
        self.raw = self.raw.dropna(subset=[col for col in self.raw.columns if col != 'Liquor FLR'+'_ramp_rate' or col != 'Total Steam Demand'+'_ramp_rate']) # remove rows with NaN values 
           
        # Filter month of shutdowns
        self.filtered = self.remove_columns_by_month(dataframe=self.raw, month_names=['June', 'November'])
        print(self.filtered)
        
        # Hampel filter - optional
        #self.filtered = self.apply_hampel_filter(col_names = [self.target_col_name], window_size=10, n_sigma=3, plot_threshold=False, keep_rows=False)
    
    def apply_hampel_filter(self, 
                            col_names, # names to apply filter to
                            window_size=10, 
                            n_sigma=5, 
                            plot_threshold=False, # plot the outliers and threshold
                            keep_rows=False, # keep or remove rows that had outliers as Hampel interpolates with the rolling median
                            ):
        
        n_rows = len(self.filtered.index)
        hampel_filtered = self.filtered.copy()
        hampel_filtered = hampel_filtered.head(n_rows)
        outlier_pos = []
        for column in col_names: 
            if self.filtered[column].dtype in [np.float64, np.int64]:
                print(column)
                result = hampel(self.filtered[column][:n_rows], window_size=window_size, n_sigma=float(n_sigma))
                hampel_filtered[column] = result.filtered_data.to_list()
                outlier_indices = result.outlier_indices # position in list than outlier was removed
                medians = result.medians
                mad_values = result.median_absolute_deviations
                thresholds = result.thresholds
                outlier_pos += outlier_indices.tolist() # save all the outlier indices

                if plot_threshold:
                    # Plot the original data with estimated standard deviations in the first subplot
                    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
                    axes[0].scatter(self.filtered.index[:n_rows], self.filtered[column][:n_rows], label='Original Data', color='b', s=1)
                    upper_bound = medians + thresholds
                    lower_bound = medians - thresholds
                    upper_bound = np.where(upper_bound > 1e3, np.nan, upper_bound) # remove extreme bounds to maintain good y-axis scaling
                    lower_bound = np.where(lower_bound < -1e3, np.nan, lower_bound)
                    axes[0].fill_between(self.filtered.index[:n_rows], upper_bound, lower_bound, color='gray', alpha=0.5, label='Median +- Threshold')
                    axes[0].set_xlabel('Data Point')
                    axes[0].set_ylabel('Value')
                    axes[0].set_title('Original Data with Bands representing Upper and Lower limits')
                    
                    for i in outlier_indices:
                        axes[0].plot(self.filtered.index[i], self.filtered[column].iloc[i], 'ro', markersize=5)  # mark removed outliers in red

                    axes[0].legend()

                    # Plot the original data in the second subplot
                    axes[1].plot(self.filtered[column][:n_rows], label='Original Data', color='g')
                    axes[1].set_xlabel('Data Point')
                    axes[1].set_ylabel('Value')
                    axes[1].set_title('Original Data')
                    axes[1].legend()

                    # Plot the filtered data in the second subplot
                    axes[2].plot(hampel_filtered[column], label='Filtered Data', color='orange')
                    axes[2].set_xlabel('Data Point')
                    axes[2].set_ylabel('Value')
                    axes[2].set_title('Filtered Data')

                    # Adjust spacing between subplots
                    #plt.tight_layout()
                    plt.show()
            
        if keep_rows is False:
            hampel_filtered = hampel_filtered.drop(hampel_filtered.index[list(OrderedDict.fromkeys(outlier_pos))]) # drop the rows that had outliers in all columns based on unique positions in outlier_pos

        return hampel_filtered 

    def apply_savgol_filter(self,
                            data,
                            window_length=10, 
                            polyorder=3,
                            derivative_threshold=115):
        
        # Use Sav-Gol filter for smoothing & calculating derivatives
        smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)
        first_deriv = savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=1)
        rolling_derivative = pd.Series(first_deriv).rolling(window=window_length, center=True).mean()
        second_deriv = savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=2)
        
      
        # Calculate location of unsteady points based on derivative threshold
        derivative_values = first_deriv
        unsteady_mask = abs(derivative_values) > derivative_threshold # true false for flagging unsteady periods based on derivative threshold
        #print(f'{sum(unsteady_mask)} unsteady data points identified ')
       
        ''' '''
        # Plot the unsteady data
        unsteady_data = self.filtered[unsteady_mask]
        plt.plot(unsteady_data.index, unsteady_data['Total NG'], 'r.', label='Unsteady Data', markersize=3)
        plt.title('Original vs. Steady vs. Unsteady Data')
        plt.xlabel('Time')
        plt.ylabel('Total NG')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.close()

        # Plot 1mill data and derivative
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(self.filtered.index, self.filtered[self.target_col_name], 'g-', label='NG Cons')
        ax1.plot(self.filtered.index, first_deriv, 'red', label='First Deriv')
        ax1.plot(self.filtered.index, rolling_derivative, 'orange', label='Rolling First Deriv')
        ax1.legend(),  ax2.legend()
        plt.show()
        
        return smoothed_data, first_deriv, second_deriv, unsteady_mask
    
    def remove_columns_by_month(self, dataframe, month_names):
        date_filtered = dataframe.loc[~dataframe.index.month_name().isin(month_names)]
        return date_filtered

    def save_processed(
            self,
            filename): # filename/identifier of data
        '''Save processed data'''
        
        # Save the filtered data
        file_to_save = self.parent_folder / "data" / "Processed {}.csv".format(filename.replace('.csv', ''))  # CWD relative file path for input files
        self.filtered.to_csv(file_to_save, index=True)
    
    def create_lagged_features(self, dataframe, lagged_col_name, lag_indices, step=1):
        for lag_index in lag_indices:
            new_col_name = f"{lagged_col_name}-{lag_index}"
            dataframe[new_col_name] = dataframe[lagged_col_name].shift(-lag_index)
            
    def group_by_production(self, data, column_names, prod_rate_col_names=[], prod_rate_bands=[]):
        medians_df = pd.DataFrame(index=[f"{lower_bound}-{upper_bound}" for lower_bound, upper_bound in zip(prod_rate_bands[:-1], prod_rate_bands[1:])], columns=column_names)
        medians_df.index.name = 'Prod Rate Range'  # Set the index name to 'Prod Rate Range'
        # Process data in medians for each column and production rate band
        for i in range(len(prod_rate_bands) - 1):
            lower_bound = prod_rate_bands[i]
            upper_bound = prod_rate_bands[i+1]
          
            # Filter by production rate band
            for column in column_names:
                if column not in prod_rate_col_names:
                    if len(prod_rate_col_names) == 1:
                        column_data = data[column][(data[prod_rate_col_names[0]] >= lower_bound) & 
                                                    (data[prod_rate_col_names[0]] <= upper_bound)].tolist()
                    elif len(prod_rate_col_names) == 2:
                        column_data = data[column][(data[prod_rate_col_names[0]] >= lower_bound) & 
                                                    (data[prod_rate_col_names[0]] <= upper_bound) &
                                                    (data[prod_rate_col_names[1]] >= lower_bound) & 
                                                    (data[prod_rate_col_names[1]] <= upper_bound)].tolist()
                    
                    # Remove bottom outliers/no-operational data and calculate the median
                    column_data.sort()  # Sort the column_data from smallest to largest
                    column_data = [value for value in column_data if pd.notnull(value) and value != '']  # Remove NaN and empty values
                    avg = np.mean(column_data)  # Calculate the average using numpy's mean function
                    column_data = [value for value in column_data if abs(value) >= abs(0.05 * avg)]  # Remove values below the threshold
                    
                    median = np.median(column_data)  
                   
                    medians_df.loc[f"{lower_bound}-{upper_bound}", column] = median    
        
        # Add a column for the sum of all values on each row
        medians_df['Total'] = medians_df.sum(axis=1)
        
        # Save medians_df to an Excel file
        medians_df.to_excel('medians.xlsx', index=True)
        
        return medians_df
        
    def inspect_data(self, df):
        # Aggregate by day and hour (or any other granularity)
        data = df.copy()
        # Heatmap - this tells us how much the value of a feature changes per hour and per day of the month
        ''' ''' 
        for column in data.columns:
            df['Day'] = df.index.day
            df['Hour'] = df.index.hour
            heatmap_data = df.groupby(['Day', 'Hour'])[column].mean().unstack()
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(heatmap_data, cmap="viridis", annot=False, cbar_kws={'label': column})
            plt.title('Heatmap of Feature Over Time')
            plt.xlabel('Hour of the Day')
            plt.ylabel('Day of the Month')
        plt.show()   
           
       
        # CUSUM - this tells us how much the current value of a feature drifts from the mean of previous values
        for column in data.columns:
            fig, ax = plt.subplots()
            # Calculate Cumulative Sum (CUSUM)
            data['Mean'] = data[column].expanding().mean()
            data['CUSUM'] = (data[column] - data['Mean']).cumsum()

            # Plot to visualize
            ax.plot(data.index, data['CUSUM'], label='CUSUM')
            ax.axhline(0, color='red', linestyle='--', label='Zero Line')
            ax.set_ylabel('Cumulative Sum')  # Modified line with superscript
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))  # Format x-axis labels as "dd-mm"
            ax.legend()
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Set ax2 to scientific numbering
            plt.show()
        
       
        '''  '''
        
        '''  ''' 
        
        # Seasonal decomposition using STL
        for column in data.columns:
            stl = STL(data[column], period=365)
            result = stl.fit()

            # Plotting the decomposition
            plt.figure(figsize=(10, 8))
            plt.subplot(4, 1, 1)
            plt.plot(df[column], label='Original', color='black')
            plt.title('Original Time Series')
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(result.trend, label='Trend', color='blue')
            plt.title('Trend Component')
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(result.seasonal, label='Seasonal', color='green')
            plt.title('Seasonal Component')
            plt.legend()

            plt.subplot(4, 1, 4)
            plt.plot(result.resid, label='Residual', color='red')
            plt.title('Residual Component')
            plt.legend()

            plt.tight_layout()
            plt.show()
        
        
        ''' '''
        # Compute EMD (using Wasserstein distance) - this tells us how much the distribution of a feature changes from week to week
        #from pyemd import emd    
        from scipy.stats import wasserstein_distance
        for column in data.columns:
            # Group data by week and calculate weekly EMD values
            weeks = data.resample("W")
            emd_values = []
            week_labels = []
            # Loop through consecutive weeks and calculate EMD

            previous_week = None
            for week_label, current_week in weeks:
                week_labels.append(week_label.strftime("%Y-%m-%d"))
                if previous_week is not None:
                    emd = wasserstein_distance(previous_week[column], current_week[column])
                    emd_values.append(emd)
                previous_week = current_week

            # Plot EMD values across weeks
            plt.figure(figsize=(10, 6))
            plt.plot(week_labels[1:], emd_values, marker="o", linestyle="-")
            plt.title(column)
            plt.xlabel("Week")
            plt.ylabel("EMD Value")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
        plt.show()
        
        
    def data_visualiton(self, data, column_names, prod_rate_col_name='', prod_rate_bands=[]):
        '''
        Visualise data
        
        Plot data to understand the behaviour of the data
        '''
        for column in column_names:
            # Filter data between speiified production rate bands
            column_data = data[column][(data[prod_rate_col_name] >= prod_rate_bands[0]) & 
                                                (data[prod_rate_col_name] <= prod_rate_bands[1])].tolist()
            plt.figure()  # Create a new figure for each column
            sns.kdeplot(column_data, label=column + ' Filtered')
            sns.kdeplot(data[column], label=column + ' Unfiltered')
            plt.legend()
        
        # box plot
        plt.show()   
        
        
      
    
