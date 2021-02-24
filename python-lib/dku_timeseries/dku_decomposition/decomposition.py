from abc import ABC, abstractmethod

import pandas as pd

from safe_logger import SafeLogger

logger = SafeLogger("Timeseries preparation plugin")


class TimeseriesDecomposition(ABC):
    """Season-Trend decomposition

    Attributes:
        dku_config(DecompositionConfig): mapping structure storing the recipe parameters
        parameters(dict): parameters formatted for the Statsmodel functions

    """

    def __init__(self, dku_config):
        self.dku_config = dku_config
        self.parameters = {}

    def fit(self, df):
        """Adds season, trend and residuals components to the input dataframe while managing long format time series

        :param df: Input dataframe. It may contain multiple time series in a long format.
        :type df: pd.DataFrame
        :return: Decomposed dataframe with season, trend and residuals columns
        :rtype: pd.DataFrame
        """
        if self.dku_config.long_format:
            decomposed_df = pd.DataFrame()
            for values, identifiers_df in df.groupby(self.dku_config.timeseries_identifiers):
                logger.info(f"Decomposing time series: Starting for the identifier {values}")
                decomposed_df = pd.concat([decomposed_df, self._decompose_df(identifiers_df)], axis=0)
                logger.info(f"Decomposing time series: Done for the identifier {values}")
        else:
            logger.info(f"Decomposing time series: Starting for the full dataset")
            decomposed_df = self._decompose_df(df)
            logger.info(f"Decomposing time series: Done for the full dataset")
        return decomposed_df

    def _decompose_df(self, timeseries_df):
        """Adds season, trend and residuals components to a dataframe which contains a unique time series

        :param timeseries_df: Dataframe which contains a unique time series
        :type timeseries_df: pd.DataFrame
        :return: Decomposed timeseries dataframe with season, trend and residuals columns
        :rtype: pd.DataFrame
        """
        time_index = timeseries_df[self.dku_config.time_column].values
        decomposed_df = timeseries_df.copy()
        for target_column in self.dku_config.target_columns:
            logger.info(f"Decomposing time series:     Starting for the target column: {target_column}")
            target_values = timeseries_df[target_column].values
            ts = self._prepare_ts(target_values, time_index)
            decomposition = self._decompose(ts)
            decomposition_columns = self._write_decomposition_columns(decomposition, timeseries_df, target_column)
            decomposed_df = pd.concat([decomposed_df, decomposition_columns], axis=1)
            logger.info(f"Decomposing time series:     Done for the target column: {target_column}")
        return decomposed_df

    def _prepare_ts(self, target_values, time_index):
        """Prepares the time series object

        :param target_values: values of a target column
        :param time_index: date range of the time series
        :return: time series
        :rtype: pd.Series
        """
        return pd.Series(target_values, index=time_index)

    @abstractmethod
    def _decompose(self, ts):
        """Estimates season, trend and residuals components of a time series

        :param ts: time series
        :type ts: pd.Series()
        :return: trend/season decomposition
        :rtype:_DecompositionResults
        """
        pass

    def _write_decomposition_columns(self, decomposition, timeseries_df, target_column):
        """Writes the trend, seasonal, residuals decomposition in a 3-column dataframe

        :param decomposition: trend/season decomposition
        :type decomposition: _DecompositionResults
        :param timeseries_df: input timeseries dataframe
        :type timeseries_df: pd.DataFrame
        :param target_column: column to decompose
        :type target_column: str
        :return: a dataframe with one column for each component
        :rtype: pd.DataFrame
        """
        new_column_names = get_component_names(target_column, timeseries_df.columns)
        decomposition_columns = pd.DataFrame(index=timeseries_df.index)
        decomposition_columns.loc[:, new_column_names["trend"]] = decomposition.trend
        decomposition_columns.loc[:, new_column_names["seasonal"]] = decomposition.seasonal
        decomposition_columns.loc[:, new_column_names["residuals"]] = decomposition.residuals
        return decomposition_columns


class _DecompositionResults:
    """Results for seasonal decomposition

    Attributes:
        trend(np.array): trend component
        seasonal(np.array): seasonal component
        residuals(np.array): residuals component
    """

    def __init__(self, trend=None, seasonal=None, residuals=None):
        self.trend = trend
        self.seasonal = seasonal
        self.residuals = residuals

    def load(self, statsmodel_results):
        self.trend = statsmodel_results.trend.values
        self.seasonal = statsmodel_results.seasonal.values
        self.residuals = statsmodel_results.resid.values


def get_component_names(target_column, columns):
    """Give a column name for each component of the decomposition

    :param target_column: column to decompose
    :type: str
    :param columns: columns of the input dataset
    :type: list
    :return: map storing the column names for each component
    :rtype: dict
    """
    new_columns_names = {}
    for component_type in ["trend", "seasonal", "residuals"]:
        new_column_name = f"{target_column}_{component_type}"
        if new_column_name not in columns:
            new_columns_names[component_type] = new_column_name
        else:
            new_columns_names[component_type] = avoid_duplicate_names(new_column_name, columns, 0)
    return new_columns_names


def avoid_duplicate_names(new_column_name, columns, suffix):
    """Adds a suffix in case of a column name collision in a recursive way

    :param new_column_name: a possible new column name
    :type: str
    :param columns: existing column names
    :type: list
    :param suffix: suffix to add to prevent collisions
    :int
    :return: a column name that prevents collisions
    :type: str
    """
    if f"{new_column_name}_{suffix}" not in columns:
        return f"{new_column_name}_{suffix}"
    else:
        suffix += 1
        return avoid_duplicate_names(new_column_name, columns, suffix)
