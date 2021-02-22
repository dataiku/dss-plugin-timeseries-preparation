from abc import ABC, abstractmethod

import pandas as pd


class TimeseriesDecomposition(ABC):
    def __init__(self, dku_config):
        self.dku_config = dku_config
        self.parameters = {}

    def fit(self, df):
        if self.dku_config.long_format:
            decomposed_df = pd.DataFrame()
            for _, identifiers_df in df.groupby(self.dku_config.timeseries_identifiers):
                decomposed_df = pd.concat([decomposed_df, self._decompose_df(identifiers_df)], axis=0)
        else:
            decomposed_df = self._decompose_df(df)
        return decomposed_df

    def _decompose_df(self, df):
        time_index = df[self.dku_config.time_column].values
        decomposed_df = df.copy()
        for target_column in self.dku_config.target_columns:
            target_values = df[target_column].values
            ts = self._prepare_ts(target_values, time_index)
            decomposition = self._decompose(ts)
            decomposition_columns = self._write_decomposition_columns(decomposition, df, target_column)
            decomposed_df = pd.concat([decomposed_df, decomposition_columns], axis=1)
        return decomposed_df

    def _prepare_ts(self, target_values, time_index):
        return pd.Series(target_values, index=time_index)

    @abstractmethod
    def _decompose(self, ts):
        pass

    def _write_decomposition_columns(self, decomposition, df, target_column):
        component_names = get_component_names(target_column, df.columns)
        decomposition_columns = pd.DataFrame(index=df.index)
        decomposition_columns.loc[:, component_names["trend"]] = decomposition.trend
        decomposition_columns.loc[:, component_names["seasonal"]] = decomposition.seasonal
        decomposition_columns.loc[:, component_names["residuals"]] = decomposition.residuals
        return decomposition_columns

    class _DecompositionResults:
        def __init__(self, trend=None, seasonal=None, residuals=None):
            self.trend = trend
            self.seasonal = seasonal
            self.residuals = residuals

        def load(self, statsmodel_results):
            self.trend = statsmodel_results.trend.values
            self.seasonal = statsmodel_results.seasonal.values
            self.residuals = statsmodel_results.resid.values


def get_component_names(target_column, columns):
    new_columns_names = {}
    for component_type in ["trend", "seasonal", "residuals"]:
        new_column_name = f"{target_column}_{component_type}"
        if new_column_name not in columns:
            new_columns_names[component_type] = new_column_name
        else:
            new_columns_names[component_type] = prevent_collision(new_column_name, columns, 0)
    return new_columns_names


def prevent_collision(new_column_name, columns, suffix):
    if f"{new_column_name}_{suffix}" not in columns:
        return f"{new_column_name}_{suffix}"
    else:
        suffix += 1
        return prevent_collision(new_column_name, columns, suffix)
