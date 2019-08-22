# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import pandas as pd
import numpy as np
import sys
import os
import random
from datetime import datetime

## Add stuff to the path to enable exec outside of DSS
plugin_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(plugin_root, 'python-lib'))
# sys.path.append(os.path.join(os.environ['DKUINSTALLDIR'], 'src/main/python/'))
sys.path.append(os.path.join(os.environ['OLDPWD'], 'src/main/python/'))

import dku_timeseries

JUST_BEFORE_SPRING_DST = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
JUST_BEFORE_FALL_DST = pd.Timestamp('20191027 02:59:00').tz_localize('CET',
                                                                     ambiguous=True)  # It's ambiguous because there are 2 instants with these dates! We select the first

TIME_COL = 'time_col'
DATA_COL = 'data_col'
GROUP_COL = 'group_col'


### Helpers to create test data, should be fixtures at some point I guess
def _make_df_with_one_col(column_data, period=pd.DateOffset(seconds=1), start_time=JUST_BEFORE_SPRING_DST):
    from datetime import datetime
    top = datetime.now()
    time = pd.date_range(start_time, None, len(column_data), period)
    top = datetime.now()
    df = pd.DataFrame({TIME_COL: time, DATA_COL: column_data})
    return df


def _make_window_aggregator_params():
    params = dku_timeseries.WindowAggregatorParams(window_width=3)
    return params


def _make_window_aggregator():
    params = _make_window_aggregator_params()
    return dku_timeseries.WindowAggregator(params)


def _make_extrema_extraction_params():
    window = _make_window_aggregator()
    params = dku_timeseries.ExtremaExtractorParams(window)
    return params


def _make_extrema_extractor():
    params = _make_extrema_extraction_params()
    return dku_timeseries.ExtremaExtractor(params)


### Test cases

def test_empty_df():
    df = _make_df_with_one_col([])
    extrema_extractor = _make_extrema_extractor()
    output_df = extrema_extractor.compute(df, TIME_COL, DATA_COL, [GROUP_COL])
    assert output_df.shape == (0, 2)


def test_single_row_df():
    df = _make_df_with_one_col([33])
    extrema_extractor = _make_extrema_extractor()
    output_df = extrema_extractor.compute(df, TIME_COL, DATA_COL, [GROUP_COL])
    assert output_df.shape == (1, 2)
    assert output_df[DATA_COL][0] == df[DATA_COL][0]


def test_incremental_df():
    length = 100
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)
    print(df.shape)
    extrema_extractor = _make_extrema_extractor()
    output_df = extrema_extractor.compute(df, TIME_COL, DATA_COL)
    assert (output_df[DATA_COL][0]) == 99
    assert (output_df[DATA_COL + '_min'][0]) == 96  # window width = 3


def test_extrema_without_neighbors():
    length = 100
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data)

    window_aggregator = dku_timeseries.WindowAggregator(dku_timeseries.WindowAggregatorParams(window_unit='milliseconds'))
    params = dku_timeseries.ExtremaExtractorParams(window_aggregator=window_aggregator)
    extrema_extractor = dku_timeseries.ExtremaExtractor(params)
    output_df = extrema_extractor.compute(df, TIME_COL, DATA_COL)
    # only have DATE_TIME col and DATA_COL of the extrema, no stats because no neighbors
    assert output_df.shape == (1, 2)
    assert output_df[DATA_COL][0] == 99


def test_group_extrema_without_neighbors():
    start_time_1 = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
    start_time_2 = pd.Timestamp('20190131 02:00:00').tz_localize('CET')
    start_time_list = [start_time_1, start_time_2]

    len1 = 100
    len2 = 10
    data1 = range(len1)
    data2 = range(len2)
    data_list = [data1, data2]

    period1 = pd.DateOffset(seconds=1)
    period2 = pd.DateOffset(seconds=1)
    period_list = [period1, period2]

    df_list = []
    for group_id, data, period, start_time in zip(range(len(data_list)), data_list, period_list, start_time_list):
        group_name = 'group_{}'.format(group_id)
        temp_df = _make_df_with_one_col(data, period=period, start_time=start_time)
        temp_df[GROUP_COL] = group_name
        df_list.append(temp_df)

    df = pd.concat(df_list, axis=0)

    window_aggregator = dku_timeseries.WindowAggregator(dku_timeseries.WindowAggregatorParams(window_unit='milliseconds'))
    params = dku_timeseries.ExtremaExtractorParams(window_aggregator=window_aggregator)
    extrema_extractor = dku_timeseries.ExtremaExtractor(params)
    output_df = extrema_extractor.compute(df, TIME_COL, DATA_COL, groupby_columns=[GROUP_COL])
    assert output_df.shape == (2, 3)
    assert np.array_equal(output_df[DATA_COL], [99, 9])


def test_incremental_group_df():
    start_time_1 = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
    start_time_2 = pd.Timestamp('20190131 02:00:00').tz_localize('CET')
    start_time_list = [start_time_1, start_time_2]

    len1 = 100
    len2 = 10
    data1 = range(len1)
    data2 = range(len2)
    data_list = [data1, data2]

    period1 = pd.DateOffset(seconds=1)
    period2 = pd.DateOffset(seconds=1)
    period_list = [period1, period2]

    df_list = []
    for group_id, data, period, start_time in zip(range(len(data_list)), data_list, period_list, start_time_list):
        group_name = 'group_{}'.format(group_id)
        temp_df = _make_df_with_one_col(data, period=period, start_time=start_time)
        temp_df[GROUP_COL] = group_name
        df_list.append(temp_df)

    df = pd.concat(df_list, axis=0)

    extrema_extractor = _make_extrema_extractor()
    output_df = extrema_extractor.compute(df, TIME_COL, DATA_COL, [GROUP_COL])

    assert output_df.groupby(GROUP_COL).get_group('group_0').data_col[0] == 99
    assert output_df.groupby(GROUP_COL).get_group('group_0')[DATA_COL + '_min'][0] == 96  # window width = 3

    assert output_df.groupby(GROUP_COL).get_group('group_1').data_col[1] == 9
    assert output_df.groupby(GROUP_COL).get_group('group_1')[DATA_COL + '_min'][1] == 6  # window width = 3
