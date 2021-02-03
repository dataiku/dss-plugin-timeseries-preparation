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
        component_names = get_component_names(target_column, df.columns)
        df[component_names["trend"]] = decomposition.trend.values
        df[component_names["seasonal"]] = decomposition.seasonal.values
        df[component_names["residuals"]] = decomposition.resid.values
        return df


def get_component_names(target_column, columns):
    new_columns_names = {}
    for component_type in ["trend", "seasonal", "residuals"]:
        new_column_name = f"{target_column}_{component_type}"
        if new_column_name not in columns:
            new_columns_names[component_type] = new_column_name
        else:
            new_columns_names[component_type] = prevent_collision(new_column_name,columns, 0)
    return new_columns_names


def prevent_collision(new_column_name,columns, suffix):
    if f"{new_column_name}_{suffix}" not in columns:
        return f"{new_column_name}_{suffix}"
    else:
        suffix += 1
        return prevent_collision(new_column_name, columns, suffix)







