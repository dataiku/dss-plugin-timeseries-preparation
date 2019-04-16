# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import logging
import re
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')

TIME_STEP_MAPPING = {
    'day': 'D',
    'hour': 'H',
    'minute': 'T',
    'second': 'S',
    'millisecond': 'ms',
}

WIN_TYPES = ['normal', 'triang', 'blackman', 'hamming', 'bartlett', 'parzen', 'gaussian', None]
TIME_UNITS = ['day', 'hour', 'minute', 'second', 'millisecond', 'row']
CLOSED_OPTIONS = ['right', 'left', 'both', 'neither']
AGGREGATIONS = ['retrieve', 'average', 'min', 'max', 'std', 'q25', 'median', 'q75', 'sum', 
                'first_order_derivative', 'second_order_derivative', 'count'] # No lag for instance, UI concern (where to put offset value)

class WindowRollerParams: #TODO better naming ?
    
    def __init__(self, 
                 time_step_size = 1, 
                 time_unit = 'second', 
                 min_period=1, 
                 closed_option='right',
                 center='left', 
                 win_type=None,
                 gaussian_std = None,
                 aggregation_types = ['mean'],
                 groupby_cols = []):
    
        self.time_step_size = time_step_size
        self.time_unit = time_unit
        
        if time_unit == 'row':
            self.window_description = self.time_step_size
        else: 
            self.window_description = str(self.time_step_size) + TIME_STEP_MAPPING.get(self.time_unit, '')
            
        self.min_period = min_period
        self.closed_option = closed_option
        self.center = center
        self.win_type = win_type
        self.gaussian_std = gaussian_std
        self.aggregation_types = aggregation_types
        self.groupby_cols = groupby_cols
        
    def check(self):
        
        if self.win_type not in WIN_TYPES:
            raise ValueError('{0} is not a valid window type. Possbile options are: {1}'.format(self.win_type, WIN_TYPES))
        
        if self.time_step_size < 0:
            raise ValueError('Time step can not be negative.')

        if self.time_unit not in TIME_UNITS:
            raise ValueError('"{0}" is not a valid unit. Possible time units are: {1}'.format(
                self.time_unit, TIME_UNITS))
            
        if self.min_period < 1:
            raise ValueError('Min period must be positive.')
            
        if self.closed_option not in CLOSED_OPTIONS:
            raise ValueError('{0} is not a valid closed option. Possible values are: {1}'.format(self.closed_option, CLOSED_OPTIONS))
           

class WindowRoller:
    
    def __init__(self, params=None):
                
        assert params is not None, "ResamplerParams instance is not specified."
        self.params = params
        self.params.check()
        
    def _check_valid_data(self, df):
        
        self.frequency = pd.infer_freq(df[~df.index.duplicated()].index[:1000])
        print('Frequency: ',self.frequency)
        # if non-equispaced + time-based window + using window, it is not possible (scipy limitation)
        if not self.frequency and self.params.time_unit != 'row' and self.params.win_type is not None:
            raise ValueError('The input dataset is not equispaced thus we can not apply window fucntions on it.') 

    def _convert_time_freq_to_row_freq(self):
        
        print(self.params.window_description)
        print(self.frequency)
        data_frequency = pd.to_timedelta(to_offset(self.frequency))
        demanded_frequency = pd.to_timedelta(to_offset(self.params.window_description))
        n = demanded_frequency/data_frequency
        print(int(round(n)))
        if n < 1:
            raise ValueError('The requested window size is smaller than the timeseries frequency. Please increase the former.')
        return int(round(n))

    def _compute_rolling_stats(self, df, raw_columns):
           
        new_df = pd.DataFrame(index=df.index)
        # compute all stats except mean and sum, does not change whether or not we have a window type
        rolling_without_window = df.rolling(window = self.params.window_description)
        
        if 'retrieve' in self.params.aggregation_types:
            new_df[raw_columns] = df[raw_columns]
        if 'min' in self.params.aggregation_types:
            col_names = ['{}_min'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].min()
        if 'max' in self.params.aggregation_types:
            col_names = ['{}_max'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].max()
        if 'q25' in self.params.aggregation_types:
            col_names = ['{}_q25'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].quantile(0.25)
        if 'median' in self.params.aggregation_types:
            col_names = ['{}_median'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].quantile(0.5)
        if 'q75' in self.params.aggregation_types:
            col_names = ['{}_q75'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].quantile(0.75)
        if 'first_order_derivative' in self.params.aggregation_types:
            col_names = ['{}_1st_derivative'.format(col) for col in raw_columns]
            new_df[col_names] = df[raw_columns].diff()
        if 'second_order_derivative' in self.params.aggregation_types:
            col_names = ['{}_2nd_derivative'.format(col) for col in raw_columns]
            new_df[col_names] = df[raw_columns].diff().diff()
        if 'std' in self.params.aggregation_types:
            col_names = ['{}_std'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].std()
            
        # compute mean, std and sum
        if self.params.win_type:
            rolling_with_window = df.rolling(self.params.window_description, 
                                             win_type=self.params.win_type)
            if 'average' in self.params.aggregation_types:
                col_names = ['{}_avg'.format(col) for col in raw_columns]
                if self.params.win_type == 'gaussian':
                    new_df[col_names] = rolling_with_window[raw_columns].mean(std=self.params.gaussian_std)
                else:
                    new_df[col_names] = rolling_with_window[raw_columns].mean()
            if 'sum' in self.params.aggregation_types:
                col_names = ['{}_sum'.format(col) for col in raw_columns]
                if self.params.win_type == 'gaussian':
                    new_df[col_names] = rolling_with_window[raw_columns].sum(std=self.params.gaussian_std)
                else:
                    new_df[col_names] = rolling_with_window[raw_columns].sum()
        else:
            if 'average' in self.params.aggregation_types:
                col_names = ['{}_avg'.format(col) for col in raw_columns]
                new_df[col_names] = rolling_without_window[raw_columns].mean()
            if 'sum' in self.params.aggregation_types:
                col_names = ['{}_sum'.format(col) for col in raw_columns]
                new_df[col_names] = rolling_without_window[raw_columns].sum()
    
        return new_df
    
    
    def compute(self, raw_df):
        
        df = raw_df.set_index('UTC').sort_index() #TODO change this!!
        self._check_valid_data(df)
        raw_columns = df.select_dtypes(include=['float', 'int']).columns.tolist()
        
        if self.frequency and self.params.time_unit != 'row' and self.params.win_type is not None:
            self.params.window_description = self._convert_time_freq_to_row_freq() #TODO: should there be a parameter ?
        
        if self.params.groupby_cols:
            grouped = df.groupby(self.params.groupby_cols)
            computed_groups = []
            for _, group in grouped:                         
                computed_df = self._compute_rolling_stats(group, raw_columns)
                computed_groups.append(computed_df) 
            final_df = pd.concat(computed_groups)
        
        else:
            final_df = self._compute_rolling_stats(group, raw_columns)
        
        final_df = final_df.rename_axis('toto').reset_index()
        return final_df
