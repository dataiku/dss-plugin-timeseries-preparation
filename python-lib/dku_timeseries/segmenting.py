# -*- coding: utf-8 -*- 
import dataiku  
import pandas as pd 
import numpy as np  
import logging  

logger = logging.getLogger(__name__)   
logging.basicConfig(level=logging.INFO, 
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')

TIME_STEP_MAPPING = {
    'day': 'D',
    'hour': 'h',
    'minute': 'm',
    'second': 's',
    'millisecond': 'ms',
    'microsecond': 'us',
    'nanosecond': 'ns',
    'row': ''
}

class SegmentExtractorParams:
    
    def __init__(self,
                 min_segment_duration_value = 0,
                 max_noise_duration_value = 0,
                 time_unit = 'second',
                 groupby_cols = None):
        
        self.min_segment_duration_value = min_segment_duration_value
        self.max_noise_duration_value = max_noise_duration_value
        self.time_unit = time_unit
        self.groupby_cols = groupby_cols
        
        if self.time_unit == 'row':
            self.min_segment_duration = self.min_segment_duration_value
            self.max_noise_duration = self.max_noise_duration_value
        else:
            self.min_segment_duration = pd.Timedelta(str(self.min_segment_duration_value) + TIME_STEP_MAPPING.get(self.time_unit, ''))
            self.max_noise_duration = pd.Timedelta(str(self.max_noise_duration_value) + TIME_STEP_MAPPING.get(self.time_unit, ''))
    
    def check(self):
        
        if self.min_segment_duration_value < 0:
            raise ValuError('Min segment duration can not be negative.')
        if self.max_noise_duration_value < 0:
            raise ValueError('Max noisy duration can not be negative.')
        if self.time_unit not in TIME_STEP_MAPPING:
            raise ValueError('{0} is not a valid time unit. Possible options are: {1}'.format(self.time_unit, TIME_STEP_MAPPING.keys()))

class SegmentExtractor:
    
    def __init__(self, params):
        assert params is not None, "SegmentExtractorParams instance is not specified."
        self.params = params
        self.params.check()
        
    def _detect_time_segment(self, df, chosen_col, lower_threshold, upper_threshold):        
        
        filtered_index = df[(df[chosen_col] >= lower_threshold) & (df[chosen_col] <= upper_threshold)].index
        
        if len(filtered_index) > 0:
            filtered_serie = filtered_index.to_series()

            d1 = filtered_serie.diff(1)
            d1 = d1.fillna(self.params.max_noise_duration)

            d2 = d1.shift(-1)
            d2 = d2.fillna(self.params.max_noise_duration)
            
            filtered = filtered_serie[d1 >= self.params.max_noise_duration].index
            filtered2 = filtered_serie[d2 >= self.params.max_noise_duration].index
            
            inds2 = np.vstack((filtered, filtered2)).T
                        
            check_segment_duration = lambda x: pd.Timedelta(x[1]-x[0]) >= self.params.min_segment_duration
            segment_indexes = filter(check_segment_duration, inds2)
            
        else: #empty
            segment_indexes = []
            
        return segment_indexes
        
    def _detect_row_segment(self, df, chosen_col, lower_threshold, upper_threshold):
        
        x = df[chosen_col].values
        # deal with NaN's (by definition, NaN's are not greater than threshold)
        x[np.isnan(x)] = np.inf
        # indices of data greater than or equal to threshold
        inds = np.nonzero((x >= lower_threshold) & (x <= upper_threshold))[0]
        if inds.size:
            # initial and final indexes of almost continuous data
            inds = np.vstack((inds[np.diff(np.hstack((-np.inf, inds))) >= self.params.max_noise_duration], \
                              inds[np.diff(np.hstack((inds, np.inf))) >= self.params.max_noise_duration])).T
            # indexes of almost continuous data longer than or equal to n_above
            segment_indexes = inds[inds[:, 1]-inds[:, 0] >= self.params.min_segment_duration, :]
        if not inds.size:
            segment_indexes = np.array([])  # standardize inds shape for output

        return segment_indexes
        
    def _detect_segment(self, raw_df, datetime_column, threshold_dict):
        
        #TODO add support for multiple threshold column
        if self.params.time_unit == 'row':
            df = raw_df.sort_values(datetime_column)
            df = df.reset_index()
        else:
            df = raw_df.set_index(datetime_column).sort_index()
        
        for chosen_column, threshold_tuple in threshold_dict.items():
            lower_threshold, upper_threshold = threshold_tuple
            if self.params.time_unit == 'row':
                segment_indexes = self._detect_row_segment(df, chosen_column, lower_threshold, upper_threshold)
            else:
                segment_indexes = self._detect_time_segment(df, chosen_column, lower_threshold, upper_threshold)            

        mask_list = []
        if len(segment_indexes) > 0:
            for start, end in segment_indexes:
                mask = (df.index >= start) & (df.index <= end)
                mask_list.append(mask)
            segment_df = df.loc[np.logical_or.reduce(mask_list)]
        else:
            segment_df = pd.DataFrame(columns=df.columns)
        
        if self.params.time_unit != 'row':
            segment_df = segment_df.rename_axis(datetime_column).reset_index()
        
        return segment_df
        
        
    def compute(self, raw_df, datetime_column, threshold_dict):
        
        if self.params.groupby_cols:
            grouped = raw_df.groupby(self.params.groupby_cols)
            segmented_groups = []
            for _, group in grouped:
                segment_df = self._detect_segment(group, datetime_column, threshold_dict)
                segmented_groups.append(segment_df)
            final_df = pd.concat(segmented_groups)        
        else:
            final_df = self._detect_segment(raw_df, datetime_column, threshold_dict)
        
        return final_df
        