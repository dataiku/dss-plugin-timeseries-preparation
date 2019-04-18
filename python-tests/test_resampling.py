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
sys.path.append(os.path.join(os.environ['DKUINSTALLDIR'], 'src/main/python/'))

import dku_timeseries

JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
JUST_BEFORE_FALL_DST = pd.Timestamp('20191027 02:59:00').tz_localize('CET', ambiguous=True) #It's ambiguous because there are 2 instants with these dates! We select the first

TIME_COL = 'time_col'
DATA_COL = 'data_col'


### Helpers to create test data, should be fixtures at some point I guess
def _make_df_with_one_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    from datetime import datetime
    top = datetime.now()
    time = pd.date_range(start_time, None, len(column_data), period)
    top = datetime.now()
    df = pd.DataFrame({TIME_COL: time, DATA_COL: column_data})
    return df

def _make_resampling_params():
    params = dku_timeseries.ResamplerParams()
    params.datetime_column = TIME_COL
    return params

def _make_resampler():
    params = _make_resampling_params()
    return dku_timeseries.Resampler(params)


### Test cases


def test_empty_df():
    df = _make_df_with_one_col([])
    resampler = _make_resampler()
    output_df = resampler.transform(df)
    assert output_df.shape == (0, 2)
    
def test_single_row_df():
    df = _make_df_with_one_col([33])
    resampler = _make_resampler()
    output_df = resampler.transform(df)
    assert output_df.shape == (1, 2)
    assert output_df[DATA_COL][0] == df[DATA_COL][0]
    
def test_identity_resampling():
    """
    Default sampling rate is 1Hz
    Since we create test data at 1Hz, default resampling should be identity
    """
    length = 100000
    print("test_identity_resampling with "+str(length)+" records")
    data = [random.random() for _ in range(length)]
    df = _make_df_with_one_col(data)
    resampler = _make_resampler()
    output_df = resampler.transform(df)
    assert output_df.shape == (length, 2)
    for x in range(1000):
        assert output_df[DATA_COL][x] == df[DATA_COL][x]

def test_double_freq_resampling():
    length = 100000
    print("test_double_freq_resampling with "+str(length)+" records")
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data, pd.DateOffset(seconds=0.5))
    resampler = _make_resampler()
    output_df = resampler.transform(df)
    print("input length:" + str(length) +" / output length: "+ str(output_df.shape[0]))
    assert output_df.shape == (length/2, 2)
    for x in range(100):
        assert output_df[DATA_COL][x] == 2*x
