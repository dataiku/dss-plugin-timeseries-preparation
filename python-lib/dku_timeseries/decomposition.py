from abc import ABC, abstractmethod
import itertools
import pandas as pd


class TimeseriesDecomposition(ABC):
    def __init__(self, recipe_config):
        self.config = recipe_config
        self.parameters = {}

    def fit(self, df):
        if self.config.long_format:
            ts_indexes = self._get_indexes_for_long_ts(df)
            decomposed_df = pd.DataFrame()
            for index in ts_indexes:
                ts_df = df.loc[index]
                decomposed_df = decomposed_df.append(self._decompose_df(ts_df))
        else:
            decomposed_df = self._decompose_df(df)
        return decomposed_df

    def _decompose_df(self,df):
        time_index = df[self.config.time_column].values
        for target_column in self.config.target_columns:
            target_values = df[target_column].values
            ts = self._prepare_ts(target_values, time_index)
            decomposition = self._decompose(ts)
            decomposed_df = self._write_decomposition(decomposition, df, target_column)
        return decomposed_df

    def _get_indexes_for_long_ts(self, df):
        identifiers = []
        for identifier_name in self.config.timeseries_identifiers:
            identifiers.append(df[identifier_name].unique())
        ts_indexes = []
        for combination in itertools.product(*identifiers):
            ts_df = df
            for i, column in enumerate(self.config.timeseries_identifiers):
                ts_df = ts_df.query(f"{column}=={combination[i]}")
            ts_indexes.append(ts_df.index)
        return ts_indexes

    def _prepare_ts(self,target_values,time_index):
        return pd.Series(target_values, index=time_index)

    @abstractmethod
    def _decompose(self, ts):
        pass

    def _write_decomposition(self, decomposition, df, target_column):
        df["{}_trend_0".format(target_column)] = decomposition.trend.values
        df["{}_seasonal_0".format(target_column)] = decomposition.seasonal.values
        df["{}_residuals_0".format(target_column)] = decomposition.resid.values
        return df



