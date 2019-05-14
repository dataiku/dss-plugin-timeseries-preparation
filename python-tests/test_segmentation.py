# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd
import sys
import os
import random
from datetime import datetime

## Add stuff to the path to enable exec outside of DSS
plugin_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(plugin_root, 'python-lib'))
#sys.path.append(os.path.join(os.environ['DKUINSTALLDIR'], 'src/main/python/'))
sys.path.append(os.path.join(os.environ['OLDPWD'], 'src/main/python/'))

import dku_timeseries

JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
JUST_BEFORE_FALL_DST = pd.Timestamp('20191027 02:59:00').tz_localize('CET', ambiguous=True) #It's ambiguous because there are 2 instants with these dates! We select the first

TIME_COL = 'time_col'
DATA_COL = 'data_col'
MIN_THRESHOLD = 0
MAX_THRESHOLD = 10
THRESHOLD_DICT = {DATA_COL: (MIN_THRESHOLD, MAX_THRESHOLD)}


### Helpers to create test data, should be fixtures at some point I guess
def _make_df_with_one_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    from datetime import datetime
    top = datetime.now()
    time = pd.date_range(start_time, None, len(column_data), period)
    top = datetime.now()
    df = pd.DataFrame({TIME_COL: time, DATA_COL: column_data})
    return df


def _make_segment_extractor_params(max_noise_duration_value=0):
    params = dku_timeseries.SegmentExtractorParams(max_noise_duration_value=max_noise_duration_value)
    return params

def _make_segment_extractor(max_noise_duration_value=0):
    params = _make_segment_extractor_params(max_noise_duration_value)
    return dku_timeseries.SegmentExtractor(params)


### Test cases

def test_empty_df():
    df = _make_df_with_one_col([])
    segment_extractor = _make_segment_extractor()
    output_df = segment_extractor.compute(df, TIME_COL, THRESHOLD_DICT)
    assert output_df.shape == (0, 2)
    
def test_single_row_df_out_range():
    df = _make_df_with_one_col([50])
    segment_extractor = _make_segment_extractor()
    output_df = segment_extractor.compute(df, TIME_COL, THRESHOLD_DICT)
    assert output_df.shape == (0, 2)

def test_single_row_df_in_range():
    df = _make_df_with_one_col([8])
    segment_extractor = _make_segment_extractor()
    output_df = segment_extractor.compute(df, TIME_COL, THRESHOLD_DICT)
    assert output_df.shape == (1, 2)
    assert output_df[DATA_COL][0] == df[DATA_COL][0]

def test_incremental_df():
    length = 1000
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data) 
    segment_extractor = _make_segment_extractor()
    output_df = segment_extractor.compute(df, TIME_COL, THRESHOLD_DICT)
    for x,y in enumerate(range(MIN_THRESHOLD, MAX_THRESHOLD)):
        assert output_df[DATA_COL][x] == y

def test_incremental_df_with_noise():
    length = 1000
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    df.loc[5:6, DATA_COL] = [50,51]
    segment_extractor = _make_segment_extractor(max_noise_duration_value=4)
    output_df = segment_extractor.compute(df, TIME_COL, THRESHOLD_DICT)
    for x,y in zip(range(len(output_df)), pd.date_range(JUST_BEFORE_SPRING_DST, None, 10, pd.DateOffset(seconds=1))):
        assert output_df[TIME_COL][x] == y
