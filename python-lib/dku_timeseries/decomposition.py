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
                decomposed_df = decomposed_df.append(self._decompose_df(identifiers_df))
        else:
            decomposed_df = self._decompose_df(df)
        return decomposed_df

    def _decompose_df(self,df):
        time_index = df[self.dku_config.time_column].values
        for target_column in self.dku_config.target_columns:
            target_values = df[target_column].values
            ts = self._prepare_ts(target_values, time_index)
            decomposition = self._decompose(ts)
            decomposed_df = self._write_decomposition(decomposition, df, target_column)
        return decomposed_df

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







