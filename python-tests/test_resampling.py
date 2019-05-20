import pandas as pd
import sys
import os
import random
from datetime import datetime
import numpy as np

## Add stuff to the path to enable exec outside of DSS
plugin_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(plugin_root, 'python-lib'))
# sys.path.append(os.path.join(os.environ['DKUINSTALLDIR'], 'src/main/python/'))
# sys.path.append(os.path.join(os.environ['OLDPWD'], 'python/'))

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


def _make_df_with_one_col_group(column_data, num_group, period=pd.DateOffset(seconds=1),
                                start_time=JUST_BEFORE_SPRING_DST):
    df_list = []
    for x in xrange(num_group):
        group_name = 'group_{}'.format(x)
        temp_df = _make_df_with_one_col(column_data, period=period)
        temp_df[GROUP_COL] = group_name
        df_list.append(temp_df)
    df = pd.concat(df_list, axis=0)
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
    output_df = resampler.transform(df, TIME_COL)
    assert output_df.shape == (0, 2)


def test_single_row_df():
    df = _make_df_with_one_col([33])
    resampler = _make_resampler()
    output_df = resampler.transform(df, TIME_COL)
    assert output_df.shape == (1, 2)
    assert output_df[DATA_COL][0] == df[DATA_COL][0]


def test_identity_resampling():
    """
    Default sampling rate is 1Hz
    Since we create test data at 1Hz, default resampling should be identity
    """
    length = 100000
    print("test_identity_resampling with " + str(length) + " records")
    data = [random.random() for _ in range(length)]
    df = _make_df_with_one_col(data)
    resampler = _make_resampler()
    output_df = resampler.transform(df, TIME_COL)
    assert output_df.shape == (length, 2)
    for x in range(1000):
        assert output_df[DATA_COL][x] == df[DATA_COL][x]


def test_half_freq_resampling():
    length = 100000
    half_length = length / 2
    print("test_double_freq_resampling with " + str(length) + " records")
    data = [x for x in range(length)]
    df = _make_df_with_one_col(data, pd.DateOffset(seconds=0.5))
    resampler = _make_resampler()
    output_df = resampler.transform(df, TIME_COL)
    print("input length:" + str(length) + " / output length: " + str(output_df.shape[0]))
    assert output_df.shape[0] - 1 == half_length
    for x in range(100):
        assert output_df[DATA_COL][x] == 2 * x


def test_identity_resampling_group():
    """
    2 groups, same frequency (1Hz), same date range
    """
    length = 100000
    num_group = 3
    print("test_identity_resampling with " + str(length) + " records")
    data = [random.random() for _ in range(length)]
    df = _make_df_with_one_col_group(data, num_group=num_group)
    resampler = _make_resampler()
    df2 = resampler.transform(df, TIME_COL, groupby_columns=GROUP_COL)
    assert len(df2) == length * num_group

    for x in xrange(num_group):
        print(x)
        df_ref = df.loc[df[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
        df_ref = df_ref.sort_values(TIME_COL)

        df_check = df2.loc[df2[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
        df_check = df_check.sort_values(TIME_COL)

        for y in xrange(1000):
            assert df_check[DATA_COL][y] == df_ref[DATA_COL][y]


def test_half_freq_resampling_group():
    """
    2 groups, same frequency (0.5Hz), same date range
    """
    length = 100000
    num_group = 3
    print("test_identity_resampling_group with " + str(2 * length) + " records")
    data = [x for x in np.arange(0, length, 0.5)]
    df = _make_df_with_one_col_group(data, num_group=num_group, period=pd.DateOffset(seconds=0.5))
    resampler = _make_resampler()
    df2 = resampler.transform(df, TIME_COL, groupby_columns=GROUP_COL)

    for x in xrange(num_group):
        print(x)
        df_ref = df.loc[df[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
        df_ref = df_ref.sort_values(TIME_COL)

        df_check = df2.loc[df2[GROUP_COL] == 'group_{}'.format(x), [TIME_COL, DATA_COL]].reset_index()
        df_check = df_check.sort_values(TIME_COL)

        for y in xrange(1000):
            assert df_check[DATA_COL][y] == 2 * df_ref[DATA_COL][y]


def test_group_inclusion_same_freq():
    """
    2 groups, same frequency, different date range: group 1 includes group 2
    """
    # 2 group, same frequency, group 1 include group 2
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

    params = dku_timeseries.ResamplerParams(extrapolation_method='interpolation')
    resampler = dku_timeseries.Resampler(params)
    output_df = resampler.transform(df, TIME_COL, groupby_columns=GROUP_COL)

    assert np.array_equal(output_df.groupby(GROUP_COL).size().values, [100, 100])

    ref_data_1 = range(0, len1)
    resample_data_1 = output_df.groupby(GROUP_COL).get_group('group_0').data_col.values
    assert np.array_equal(resample_data_1, ref_data_1)

    start_value = -(start_time_2 - start_time_1).seconds
    ref_data_2 = range(start_value, start_value + len1)
    resample_data_2 = output_df.groupby(GROUP_COL).get_group('group_1').data_col.values
    assert np.array_equal(resample_data_2, ref_data_2)


def test_group_inclusion_different_freq():
    """
    2 groups, different frequency, different date range: group 1 includes group 2
    """
    start_time_1 = pd.Timestamp('20190131 01:59:00').tz_localize('CET')
    start_time_2 = pd.Timestamp('20190131 02:00:00').tz_localize('CET')
    start_time_list = [start_time_1, start_time_2]

    len1 = 100
    len2 = 10
    data1 = range(len1)
    data2 = range(len2)
    data_list = [data1, data2]

    period1 = pd.DateOffset(seconds=2)
    period2 = pd.DateOffset(seconds=1)
    period_list = [period1, period2]

    df_list = []
    for group_id, data, period, start_time in zip(range(len(data_list)), data_list, period_list, start_time_list):
        group_name = 'group_{}'.format(group_id)
        temp_df = _make_df_with_one_col(data, period=period, start_time=start_time)
        temp_df[GROUP_COL] = group_name
        df_list.append(temp_df)

    df = pd.concat(df_list, axis=0)

    params = dku_timeseries.ResamplerParams(extrapolation_method='interpolation')
    resampler = dku_timeseries.Resampler(params)
    output_df = resampler.transform(df, TIME_COL, groupby_columns=GROUP_COL)

    assert np.array_equal(output_df.groupby(GROUP_COL).size().values, [199, 199])

    ref_data_0 = np.linspace(0, 99, num=199)
    resample_data_0 = output_df.groupby(GROUP_COL).get_group('group_0').data_col.values
    assert np.array_equal(resample_data_0, ref_data_0)

    start_value = -(start_time_2 - start_time_1).seconds
    ref_data_1 = np.linspace(start_value, start_value + 198, num=199)
    resample_data_1 = output_df.groupby(GROUP_COL).get_group('group_1').data_col.values
    assert np.array_equal(resample_data_1, ref_data_1)
