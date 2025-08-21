'''
__author__ = 'Isaac Severinsen'
__credits__ = 

Class for extracting steady state regions from a time series dataset.
'''

import pandas as pd
import numpy as np
from sklearn.cluster import k_means
from scipy.stats import gaussian_kde
from scipy.spatial.distance import mahalanobis, pdist, squareform
from numba import jit
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scienceplots
parent_folder = Path(__file__).parent # location of the folder containing this file relative to your C drive

class SteadyState:
    def __init__(self, 
                 data, 
                 rolling_width,  # the number of points to consider with the rolling calculation
                 pct, # [%] percentile to consider as stable
                 cols_to_steady): # column names to consider for steadying
       
        self.steady_state(data, rolling_width, pct, cols_to_steady)
        
    def steady_state(self, data, rolling_width, pct, cols_to_steady):
        '''
        unit_labels = pd.read_excel(file_to_open, header=None, usecols = usecols, nrows=data_start-1,  engine='openpyxl') # read in column labels
        unit_labels.columns = unit_labels.iloc[header_start - 1]
        unit_labels = unit_labels.drop(header_start - 1, axis=0) # drop the row as its now in header
        unit_labels = unit_labels.set_index('Descr')
        '''
        # Roling window calculation
        self.original_data = data.copy()  # Retain the original data
        df = data.copy()
        N = len(df)
        df2 = data.copy()
        for i in cols_to_steady:
            df2[f'rel_{i}'] = df[i].rolling(rolling_width, center=True).apply(
                self.relativevar, engine='numba', raw=True, engine_kwargs=dict(nopython=True)
            )
            try:
                thresh = np.percentile(df2[f'rel_{i}'].dropna().to_numpy(), pct)
                bool_arr = (df2[f'rel_{i}'] < thresh)
            except:
                pass
            try:
                bool_arr_comb = np.logical_and(bool_arr_comb, bool_arr)
            except:
                bool_arr_comb = bool_arr.copy()
            print(bool_arr_comb.sum())

        df['stable'] = bool_arr_comb
        stable_data = df[df['stable']]
        print('stable ', round((bool_arr_comb.sum() / len(df2)) * 100, 2), '% of the time')

        '''
        # Apply clustering only to `cols_to_steady`
        retain = 0.9
        reduce = 0.5
        clustered_data = self.cluster(stable_data, cols_to_steady, retain, reduce)
        print(len(clustered_data), 'stable datapoints')
        print(round(len(clustered_data) * 100 / N, 2), '% of original dataset')
        
        # Merge clustered results with the original data
        df = self.original_data.loc[clustered_data.index].copy()
        df.update(clustered_data)
        '''
        # Optional: Remove temporary columns
        columns_to_remove = [col for col in stable_data.columns if col.startswith('rel_') or col == 'stable']
        stable_data.drop(columns=columns_to_remove, inplace=True)

        '''
        # Insert unit label into dataframe
        insert_index = 0  # insert unit labels after header row
        selected_row_numbers = [0,1]  # row numbers to select
        selected_rows = unit_labels.iloc[selected_row_numbers]
        df = pd.concat([df.iloc[:insert_index], selected_rows, df.iloc[insert_index:]], ignore_index=False) # use concat() to insert the new rows into the original DataFrame, ignore the time row index

        filename = filename_to_save + '_stabled.csv'
        file_to_save = parent_folder / "stabled" /  filename  # CWD relative file path for input files
        df.to_csv(file_to_save)
        print(df)
        '''
        self.steadied = stable_data
        #self.plot_steady_regions(data, bool_arr_comb, cols_to_steady)
        
        

    def rarity(self,df, mean_list_dist, std_list_dist):
        scaler = lambda x: (x - x.min())/np.ptp(x)
        kde_rarity = 1-scaler(self.kde_MDdist(df[mean_list_dist].to_numpy(),df[std_list_dist].to_numpy()))

        cov = np.cov(df[mean_list_dist].to_numpy(), rowvar=False)
        mahal_list = squareform(pdist(df[mean_list_dist].to_numpy(), 'mahalanobis', VI = cov))

        mahal_rarity = scaler(np.sum(mahal_list,axis=1))
        df['rarity'] = 0.8*kde_rarity + 0.2*mahal_rarity

        return df

    def kde_MDdist(self, numdf,stddf):    # could try scikitlearns version for increased speed?
        '''
        This function calculates the multi-dimensional kernel density estimate of the data.
        Effectively a gaussian distribution is established for each point in all dimensions which is spherical
        i.e. covariance = 1

        The weighting of the data is by the value of the standard deviation summed in all dimensions.
        The bandwidth of the kernel is determine with scott's method divided by a scalar.

        is a helper function that scales an array/list of values to a 0 to 1 scale.
        In addition to the scaling of the variables themselves before ranking occurs this is used
        to combine different ranking metrics with specified weightings.
        Function Inputs:
        - numdf   : a numpy array of variables' means
        - stddf   : a numpy array of variables' standard deviations

        Fuction Outputs:
        - kdeMD_list   : A 1D list of scored samples to be used in anchor identification.
        '''
        # scaler = lambda x: (x - x.min())/np.ptp(x)
        weight_list = np.mean(stddf,axis = 1) # the scaled standard deviation summed along all axis
        # general the kernel estimate using the data, bandwidth and weights
        kernel = gaussian_kde(numdf.T, bw_method = 'scott')#,weights = 1-weight_list)
        # adjust the bandwidth by a scalar factor.
        kernel.set_bandwidth(kernel.factor/1)
        # evaluate the kernel at all data points, then scale and invert the data.
    #     return 1 - scaler(kernel.evaluate(numdf.T))
        return kernel.evaluate(numdf.T)
        
    def cluster(self, df, cols_to_steady, retain, reduction):
        col_mean = [col + '_mean' for col in cols_to_steady]
        col_std = [col + '_std' for col in cols_to_steady]

        # Rename only `cols_to_steady` for clustering
        rename_dict = {col: col + '_mean' for col in cols_to_steady}
        df = df.rename(columns=rename_dict)

        # Calculate standard deviations for `cols_to_steady`
        for mean, std in zip(col_mean, col_std):
            df[std] = 0.001 * df[mean].std()

        # Filter and perform rarity scoring
        input_list_mean = [col for col in col_mean]
        input_list_std = [col for col in col_std]

        df = self.rarity(df, input_list_mean, input_list_std)

        # Apply clustering on `cols_to_steady`
        mean_list = col_mean
        std_list = col_std

        N_old = len(df)
        rare = np.linspace(df['rarity'].min(), df['rarity'].max(), 1000)
        arr = [len(df[df['rarity'] > i]) for i in rare]
        rarity_threshold = rare[np.argmax(arr < np.ones(len(arr)) * retain * N_old)]

        df_save = df[df['rarity'] > rarity_threshold]
        N_new = int(N_old * (1 - retain) * (1 - reduction))

        X = df[df['rarity'] < rarity_threshold][mean_list].to_numpy()
        Y = k_means(X, N_new, verbose=True)
        df2 = pd.DataFrame(data=Y[0], columns=mean_list)
        df2[std_list] = 0
        df2['time_mean'] = 0
        df2['time_width'] = 0

        # Add cluster information to DataFrame
        unq = np.unique(Y[1])
        for i in unq:
            for ii, jj in enumerate(df2.columns):
                if 'time_mean' in jj:
                    time_list = df.index[np.where(Y[1] == i)]
                    df2.loc[i, 'time_mean'] = time_list.mean()
                    df2.loc[i, 'time_width'] = (time_list.max() - time_list.min()).total_seconds()
                elif '_std' in jj:
                    range_std = df[mean_list].iloc[np.where(Y[1] == i)].std()[jj[:-4] + '_mean']
                    inherent_std = df[std_list].iloc[np.where(Y[1] == i)].mean()[jj[:-4] + '_std']
                    df2.iloc[i, ii] = np.nanmax([range_std, inherent_std])

        # Snap time_mean to the nearest 5-minute interval
        valid_timestamps = df.index
        snapped_indices = valid_timestamps[
            valid_timestamps.get_indexer(pd.to_datetime(df2['time_mean']), method='nearest')
        ]
        df2.index = snapped_indices
        df2 = df2.drop(['time_mean', 'time_width'], axis=1)  # Optional: Drop time columns
        df2 = df2.sort_index()

        # Handle duplicate indices by aggregating their values
        if df2.index.duplicated().any():
            df2 = df2.groupby(df2.index).mean()

        df_save = df_save.sort_index()

        # Combine clustered data with saved rare data
        df_final = pd.concat([df2, df_save[mean_list + std_list]]).sort_index()
        return df_final
        

    @jit(nopython=True)
    def relativevar(x):
        mu = np.mean(x)
        if mu < 1e-8:
            return np.nan
        else:
            return np.std(x)/np.mean(x)
        
    def plot_steady_regions(self, data, steady_mask, cols_to_steady, title_suffix=""):
        plt.style.use('science')
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Constantia"],
            "font.size": 12,
            "figure.figsize": (8.27, 4),
            #"figure.dpi": 500
            })  # Set the figure size to A4 dimensions and DPI to 1000
        
        for col in cols_to_steady:
            fig, ax = plt.subplots()
            ax.plot(data.index, data[col] * 42/1000, label='Unsteady Periods', color='red', alpha=0.6)  # Multiply y-axis values by 42
            
            # Highlight steady state regions
            ax.plot(data.index[steady_mask], 
                    data[col][steady_mask] * 42/1000,  # Multiply y-axis values by 42
                    label='Steady Periods', 
                    color='green', 
                    )
            
            #ax.set_title(f'{col} with Steady-State Regions {title_suffix}')
            #ax.set_xlabel('Time')
            ax.set_ylabel('Total Gas Consumption (GJ/h)')  # Modified line with superscript
            
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))  # Format x-axis labels as "dd-mm"
            #plt.tight_layout()
            plt.show()