# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import logging
import re
from pandas.tseries.frequencies import to_offset
import math

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')


TIME_STEP_MAPPING = {
    'year': 'A',
    'month': 'M',
    'week': 'W',
    'day': 'D',
    'hour': 'H',
    'minute': 'T',
    'second': 'S',
    'millisecond': 'L',
    'microsecond': 'us',
    'nanosecond': 'ns'
}

WINDOW_TYPES = ['triang', 'blackman', 'hamming', 'barlett', 'parzen', 'gaussian', None]
WINDOW_UNITS = list(TIME_STEP_MAPPING.keys()) + ['row']
CLOSED_OPTIONS = ['right', 'left', 'both', 'neither']
AGGREGATIONS = ['retrieve', 'average', 'min', 'max', 'std', 'q25', 'median', 'q75', 'sum', 
                'first_order_derivative', 'second_order_derivative', 'count'] # No lag for instance, UI concern (where to put offset value)

class WindowRollerParams: #TODO better naming ?
    
    def __init__(self, 
                 window_width = 1, 
                 window_unit = 'second', 
                 min_period=1, 
                 closed_option='left',
                 center=False, 
                 window_type=None,
                 gaussian_std = None,
                 aggregation_types = AGGREGATIONS,
                 groupby_cols = None):
    
        self.window_width = window_width
        self.window_unit = window_unit
        
        if window_unit == 'row':
            self.window_description = self.window_width
        else: 
            self.window_description = str(self.window_width) + TIME_STEP_MAPPING.get(self.window_unit, '')
            
        self.min_period = min_period
        self.closed_option = closed_option
        self.center = center
        self.window_type = window_type
        self.gaussian_std = gaussian_std
        self.aggregation_types = aggregation_types
        self.groupby_cols = groupby_cols
        
    def check(self):

        if self.window_type not in WINDOW_TYPES:
            raise ValueError('{0} is not a valid window type. Possbile options are: {1}'.format(self.window_type, WINDOW_TYPES))
        
        if self.window_width < 0:
            raise ValueError('Window width can not be negative.')

        if self.window_unit not in WINDOW_UNITS:
            raise ValueError('"{0}" is not a valid unit. Possible window units are: {1}'.format(
                self.window_unit, WINDOW_UNITS))
            
        if self.min_period < 1:
            raise ValueError('Min period must be positive.')
            
        if self.closed_option not in CLOSED_OPTIONS:
            raise ValueError('"{0}" is not a valid closed option. Possible values are: {1}'.format(self.closed_option, CLOSED_OPTIONS))
           

class WindowRoller:
    
    def __init__(self, params=None):
                
        assert params is not None, "WindowRollerParams instance is not specified."
        self.params = params
        self.params.check()
        
    def _check_valid_data(self, df):
        
        # if non-equispaced + time-based window + using window, it is not possible (scipy limitation)
        if not self.frequency and self.params.window_unit != 'row' and self.params.window_type is not None:
            raise ValueError('The input dataset is not equispaced thus we can not apply window fucntions on it.') 

    def _convert_time_freq_to_row_freq(self):
        
        data_frequency = pd.to_timedelta(to_offset(self.frequency))
        demanded_frequency = pd.to_timedelta(to_offset(self.params.window_description))
        n = demanded_frequency/data_frequency
        if n < 1:
            raise ValueError('The requested window width ({0}) is smaller than the timeseries frequency ({1}). Please increase the former.'.format(data_frequency, demanded_frequency))
        return int(math.ceil(n)) # always round up so that we dont miss any data

    def get_window_date_offset(self):
        
        if self.params.window_unit == 'year':
            return pd.DateOffset(years=self.params.window_width)
        elif self.params.window_unit == 'month':
            return pd.DateOffset(months=self.params.window_width)
        elif self.params.window_unit == 'week':
            return pd.DateOffset(weeks=self.params.window_width)
        elif self.params.window_unit == 'day':
            return pd.DateOffset(days=self.params.window_width)
        elif self.params.window_unit == 'hour':
            return pd.DateOffset(hours=self.params.window_width)
        elif self.params.window_unit == 'minute':
            return pd.DateOffset(minutes=self.params.window_width)
        elif self.params.window_unit == 'second':
            return pd.DateOffset(seconds=self.params.window_width)
        elif self.params.window_unit == 'microsecond':
            return pd.DateOffset(microseconds=self.params.window_width)
        elif self.params.window_unit == 'nanosecond':
            return pd.DateOffset(nanoseconds=self.params.window_width)
        
    
    def _compute_rolling_stats(self, df, raw_columns):
           
        new_df = pd.DataFrame(index=df.index)

        if self.params.groupby_cols:
        	new_df[self.params.groupby_cols] = df[self.params.groupby_cols]

        # compute all stats except mean and sum, does not change whether or not we have a window type
        if self.params.window_unit == 'row': # for now `closed` is only implemented for time-based window
            rolling_without_window = df.rolling(window = self.params.window_description)
        else:
            rolling_without_window = df.rolling(window = self.params.window_description,
                                                closed = self.params.closed_option)


        if 'retrieve' in self.params.aggregation_types:
            new_df[raw_columns] = df[raw_columns]
    	if 'min' in self.params.aggregation_types:
            col_names = ['{}_min'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].apply(min) #raw=True
        if 'max' in self.params.aggregation_types:
            col_names = ['{}_max'.format(col) for col in raw_columns]
            new_df[col_names] = rolling_without_window[raw_columns].apply(max)
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
        if self.params.window_type:
            if self.params.closed_option == 'left':
                shifted_df = df.shift(1)
            else:
                shifted_df = df
        
            if self.params.window_unit == 'row': # 
                rolling_with_window = shifted_df.rolling(window = self.params.window_description,
	                                             		 win_type=self.params.window_type)
            else:
        		rolling_with_window = shifted_df.rolling(window = self.params.window_description_in_row, 
			                                             win_type=self.params.window_type,
			                                             closed = self.params.closed_option)

            if 'average' in self.params.aggregation_types:
                col_names = ['{}_avg'.format(col) for col in raw_columns]
                if self.params.window_type == 'gaussian':
                    new_df[col_names] = rolling_with_window[raw_columns].mean(std=self.params.gaussian_std)
                else:
                    new_df[col_names] = rolling_with_window[raw_columns].mean()
            if 'sum' in self.params.aggregation_types:
                col_names = ['{}_sum'.format(col) for col in raw_columns]
                if self.params.window_type == 'gaussian':
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
    
    
    def compute(self, raw_df, datetime_column=None):
        
        if datetime_column: 
            df = raw_df.set_index(datetime_column).sort_index() 
        else: # otherwise we suppose that the input df already has Datetime Index
            df = raw_df.copy()
        self.frequency = pd.infer_freq(df[~df.index.duplicated()].index[:1000])
        logger.info('Timeseries frequency: ',self.frequency)

        if self.params.window_unit != 'row':
        	window_width_in_row = self._convert_time_freq_to_row_freq() 

        raw_columns = df.select_dtypes(include=['float', 'int']).columns.tolist()
        self._check_valid_data(df)
 
        if self.frequency and self.params.window_unit != 'row' and self.params.window_type is not None:
            self.params.window_description_in_row = window_width_in_row
        else:
        	self.params.window_description_in_row = None
        
        if self.params.groupby_cols:
            grouped = df.groupby(self.params.groupby_cols)
            computed_groups = []
            for _, group in grouped: 
                computed_df = self._compute_rolling_stats(group, raw_columns)
                computed_groups.append(computed_df) 
            final_df = pd.concat(computed_groups)
        else:
            final_df = self._compute_rolling_stats(df, raw_columns)
        
        if datetime_column:
            final_df = final_df.rename_axis(datetime_column).reset_index()

        return final_df