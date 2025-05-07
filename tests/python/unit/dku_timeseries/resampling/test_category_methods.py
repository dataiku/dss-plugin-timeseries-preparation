import math

import numpy as np
import pandas as pd
import pytest

from dku_timeseries import Resampler
from recipe_config_loading import get_resampling_params

@pytest.fixture
def columns():
    class COLUMNS:
        date = "Date"
        category = "categorical"
        data = "value1"
    return COLUMNS

@pytest.fixture
def df(columns):
    co2 = [315.58, 316.39, 316.79, 316.2]
    categorical = ["first", "first", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq="M")
    df = pd.DataFrame.from_dict(
        {columns.data: co2, "value2": co2, columns.category: categorical, columns.date: time_index})
    return df


@pytest.fixture
def df2(columns):
    dates = np.array(['2013-01-01T00:00:00.000000000', '2013-01-03T00:00:00.000000000',
                      '2013-01-04T00:00:00.000000000', '2013-01-05T00:00:00.000000000',
                      '2013-01-06T00:00:00.000000000'], dtype='datetime64[ns]')
    holiday = np.array([0, 1, 2, 3, 4])
    sales_1 = np.array([13, 14, 13, 10, 12])
    categorical = ["first", "first", "second", "second", "third"]
    df2 = pd.DataFrame.from_dict({columns.date: dates, "holiday": holiday, columns.data: sales_1, columns.category: categorical})
    return df2


@pytest.fixture
def df3(columns):
    co2 = [315.58, 316.39, 316.79, 316.2]
    categorical = ["first", "second", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq="M")
    df = pd.DataFrame.from_dict(
        {columns.data: co2, "value2": co2, columns.category: categorical, columns.date: time_index})
    return df


@pytest.fixture
def missing_row_df(columns):
    co2 = [315.58, 316.39, 316.79, 316.2]
    categorical = [np.nan, "second", "second", "second"]
    time_index = pd.date_range("1-1-1959", periods=4, freq="M")
    df = pd.DataFrame.from_dict(
        {columns.data: co2, "value2": co2, columns.category: categorical, columns.date: time_index})
    return df


@pytest.fixture
def long_df(columns):
    co2 = [315.58, 316.39, 316.79, 316.2]
    country = [0, 0, 1, 1]
    categorical = ["first", "second", "third", "fourth"]
    time_index = pd.date_range("1-1-1959", periods=2, freq="M").append(pd.date_range("1-1-1959", periods=2, freq="M"))
    long_df = pd.DataFrame.from_dict(
        {columns.data: co2, "value2": co2, "country": country, columns.category: categorical, columns.date: time_index})
    return long_df


@pytest.fixture
def long_df_mode(columns):
    co2 = [315.58, 316.39, 300, 316.79, 316.2, 390]
    country = [0, 0, 0, 1, 1, 1]
    categorical = ["first", "first", "second", "third", "fourth", "fourth"]
    time_index = pd.date_range("1-1-1959", periods=3, freq="M").append(pd.date_range("1-1-1959", periods=3, freq="M"))
    long_df = pd.DataFrame.from_dict(
        {columns.data: co2, "value2": co2, "country": country, columns.category: categorical, columns.date: time_index})
    return long_df


@pytest.fixture
def df_multiple_dates(columns):
    dates = np.array(['2013-01-01T00:00:00.000000000', '2013-01-03T00:00:00.000000000',
                      '2013-01-04T00:00:00.000000000', '2013-01-05T00:00:00.000000000',
                      '2013-01-06T00:00:00.000000000'], dtype='datetime64[ns]')
    holiday = np.array([0, 1, 2, 3, 4])
    sales_1 = np.array([13, 14, 13, 10, 12])
    categorical = ["first", "first", "second", "second", "third"]
    df2 = pd.DataFrame.from_dict({columns.date: dates, "holiday": holiday, "sales_1": sales_1, columns.category: categorical, "date2": dates})
    return df2


@pytest.fixture
def bool_df(columns):
    dates = np.array(['2013-01-01T00:00:00.000000000', '2013-01-03T00:00:00.000000000',
                      '2013-01-04T00:00:00.000000000', '2013-01-05T00:00:00.000000000',
                      '2013-01-06T00:00:00.000000000'], dtype='datetime64[ns]')
    holiday = np.array([0, 1, 2, 3, 4])
    sales_1 = np.array([13, 14, 13, 10, 12])
    categorical = [True, True, False, False, False]
    df = pd.DataFrame.from_dict({columns.date: dates, "holiday": holiday, "sales_1": sales_1, columns.category: categorical})
    return df


@pytest.fixture
def config(columns):
    config = {u'clip_end': 0, u'constant_value': 0, u'extrapolation_method': u'clip', u'shift': 0, u'time_unit_end_of_week': u'SUN',
              u'datetime_column': columns.date, u'advanced_activated': False, u"groupby_columns": [], u'time_unit': u'weeks', u'clip_start': 0,
              u'time_step': 2, "category_imputation_method": "empty", u'interpolation_method': u'linear'}
    return config


class TestCategoryMethods:
    def test_extrapolation(self, df, config, columns):
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df, columns.date)
        assert output_df.loc[7, columns.data] == 316.2
        assert math.isnan(output_df.loc[7, columns.category])

        config.pop("category_imputation_method")
        resampler = Resampler(params)
        output_df = resampler.transform(df, columns.date)
        assert output_df.loc[7, columns.data] == 316.2
        assert math.isnan(output_df.loc[7, columns.category])

        config["extrapolation_method"] = "none"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df, columns.date)
        assert math.isnan(output_df.loc[6, columns.category])
        category_results = np.array(output_df[columns.category].values, dtype=np.float64)
        assert np.isnan(category_results).all()

        config["extrapolation_method"] = "interpolation"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df, columns.date)
        assert np.round(output_df.loc[7, columns.data], 3) == 316.003
        assert math.isnan(output_df.loc[7, columns.category])

    def test_empty_filling(self, df2, config, columns):
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df2, columns.date)
        assert output_df.loc[0, columns.category] == "first"
        assert math.isnan(output_df.loc[1, columns.category])
        assert math.isnan(output_df.loc[2, columns.category])
        assert output_df.loc[6, columns.category] == "second"
        assert math.isnan(output_df.loc[7, columns.category])

    def test_constant_value_filling(self, df2, config, columns):
        config["category_imputation_method"] = "constant"
        config["category_constant_value"] = "myvalue"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df2, columns.date)
        assert output_df.loc[0, columns.category] == "first"
        assert output_df.loc[1, columns.category] == "myvalue"
        assert output_df.loc[2, columns.category] == "myvalue"
        assert output_df.loc[6, columns.category] == "second"
        assert output_df.loc[7, columns.category] == "myvalue"

    def test_missing_constant(self, df2, config, columns):
        config["category_imputation_method"] = "constant"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df2, columns.date)
        np.testing.assert_array_equal(output_df.categorical.values, np.array(['first', '', '', '', 'first', '', 'second', '', 'second', '', 'third']))

    def test_previous_filling(self, df2, config, columns):
        config["category_imputation_method"] = "previous"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df2, columns.date)
        np.testing.assert_array_equal(output_df.categorical.values,
                                      np.array(['first', 'first', 'first', 'first', 'first', 'first', 'second', 'second', 'second', 'second', 'third']))

    def test_next_filling(self, df2, config, columns):
        config["category_imputation_method"] = "next"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df2, columns.date)
        assert output_df.loc[0, columns.category] == "first"
        assert output_df.loc[1, columns.category] == "first"
        assert output_df.loc[3, columns.category] == "first"
        assert output_df.loc[5, columns.category] == "second"
        assert output_df.loc[9, columns.category] == "third"

    def test_no_category_values(self, df, config,columns):
        config["category_imputation_method"] = "previous"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df_first = resampler.transform(df, columns.date)
        np.testing.assert_array_equal(output_df_first.categorical.values,
                                      np.array(['first', 'first', 'first', 'first', 'first', 'second', 'second', 'second']))

        config["category_imputation_method"] = "empty"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df_empty = resampler.transform(df, columns.date)
        assert math.isnan(output_df_empty.loc[0, columns.category])

    def test_previous_filling_long_format(self, long_df, config,columns):
        config["category_imputation_method"] = "previous"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(long_df, columns.date, groupby_columns=["country"])
        expected_dates = pd.DatetimeIndex(['1959-02-01T00:00:00.000000000', '1959-02-08T00:00:00.000000000',
                                           '1959-02-15T00:00:00.000000000', '1959-02-22T00:00:00.000000000',
                                           '1959-03-01T00:00:00.000000000', '1959-02-01T00:00:00.000000000',
                                           '1959-02-08T00:00:00.000000000', '1959-02-15T00:00:00.000000000',
                                           '1959-02-22T00:00:00.000000000', '1959-03-01T00:00:00.000000000'])
        np.testing.assert_array_equal(output_df[columns.date].values, expected_dates)
        expected_categorical = np.array(['first', 'first', 'first', 'first', 'second', 'third', 'third', 'third', 'third', 'fourth'])
        np.testing.assert_array_equal(output_df.categorical.values, expected_categorical)

    def test_next_filling_long_format(self, long_df, config,columns):
        config["category_imputation_method"] = "next"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(long_df, columns.date, groupby_columns=["country"])
        assert math.isnan(output_df.loc[4, columns.category])
        assert output_df.loc[3, columns.category] == "second"

    def test_clip_filling(self, long_df, config,columns):
        config["category_imputation_method"] = "clip"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(long_df, columns.date, groupby_columns=["country"])
        assert output_df.loc[3, columns.category] == "first"

    def test_mode_filling(self, df3, config,columns):
        config["category_imputation_method"] = "mode"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df3, columns.date)
        assert np.all(output_df.categorical.values == "second")

    def test_mode_filling_long_format(self, long_df_mode, config,columns):
        config["category_imputation_method"] = "mode"
        config["time_unit"] = "weeks"
        config["time_step"] = 1
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(long_df_mode, columns.date, groupby_columns=["country"])
        assert np.all(output_df.loc[output_df.country == 0, columns.category].values == "first")
        assert np.all(output_df.loc[output_df.country == 1, columns.category].values == "fourth")

    def test_missing_categorical(self, missing_row_df, config,columns):
        config["time_unit"] = "weeks"
        config["time_step"] = 12
        config["category_imputation_method"] = "clip"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(missing_row_df, columns.date)
        assert np.all(output_df.categorical.values == "second")

        config["category_imputation_method"] = "previous"
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(missing_row_df, columns.date)
        assert math.isnan(output_df.loc[0, columns.category])
        assert np.all(output_df.loc[1:, columns.category].values == "second")

    def test_df_multiple_dates(self, df_multiple_dates, config,columns):
        config["category_imputation_method"] = "previous"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(df_multiple_dates, columns.date)
        np.testing.assert_array_equal(pd.to_datetime(output_df['date2']).map(lambda s: s.strftime('%Y-%m-%d')), np.array(["2013-01-01", "2013-01-01", "2013-01-01", "2013-01-01", "2013-01-03", "2013-01-03", "2013-01-04", "2013-01-04", "2013-01-05", "2013-01-05", "2013-01-06"]))

    def test_bool_column(self, bool_df, config,columns):
        config["category_imputation_method"] = "previous"
        config["time_unit"] = "hours"
        config["time_step"] = 12
        params = get_resampling_params(config)
        resampler = Resampler(params)
        output_df = resampler.transform(bool_df, columns.date)
        np.testing.assert_array_equal(output_df.categorical.values, np.array([True, True, True, True, True, True, False, False, False, False, False]))

    def test_no_categorical_impute(self, df, config,columns):
        config.pop("category_imputation_method")
        params_no_impute = get_resampling_params(config)
        resampler_no_impute = Resampler(params_no_impute)
        no_impute_df = resampler_no_impute.transform(df, "Date")
        assert pd.isnull(no_impute_df[columns.category].values).all()

        config["category_imputation_method"] = "empty"
        params_with_impute =  get_resampling_params(config)
        resampler_with_impute = Resampler(params_with_impute)
        impute_df = resampler_with_impute.transform(df, "Date")
        assert pd.isnull(impute_df[columns.category].values).all()
