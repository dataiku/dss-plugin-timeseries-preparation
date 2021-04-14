import numpy as np
from statsmodels.tsa.seasonal import STL

from dku_timeseries.dku_decomposition.decomposition import TimeseriesDecomposition, _DecompositionResults


class STLDecomposition(TimeseriesDecomposition):
    """Season-Trend decomposition using Loess

    Attributes:
        dku_config(DecompositionConfig): mapping structure storing the recipe parameters
        parameters(dict): parameters formatted for the Statsmodel functions

    """

    def __init__(self, dku_config):
        super().__init__(dku_config)
        self.parameters = format_parameters(dku_config)

    def _decompose(self, ts):
        if self.dku_config.model == "multiplicative":
            self.parameters["endog"] = np.log(ts)
            stl = STL(**self.parameters)
            statsmodel_results = stl.fit()
            trend = np.exp(statsmodel_results.trend.values)
            seasonal = np.exp(statsmodel_results.seasonal.values)
            residuals = np.exp(statsmodel_results.resid.values)
            decomposition = _DecompositionResults(trend=trend, seasonal=seasonal, residuals=residuals)
        elif self.dku_config.model == "additive":
            self.parameters["endog"] = ts
            stl = STL(**self.parameters)
            statsmodel_results = stl.fit()
            decomposition = _DecompositionResults()
            decomposition.load(statsmodel_results)
        return decomposition


def format_parameters(dku_config):
    parameters = {"period": dku_config.period}
    if dku_config.advanced:
        parameters["seasonal"] = dku_config.seasonal
        parameters["robust"] = dku_config.robust_stl
        additional_parameters = dku_config.get_param("additional_parameters_STL")
        if additional_parameters:
            parameters.update(dku_config.additional_parameters_STL)
    return parameters
