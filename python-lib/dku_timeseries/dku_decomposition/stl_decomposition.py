import numpy as np
from statsmodels.tsa.seasonal import STL

from dku_timeseries.dku_decomposition.decomposition import TimeseriesDecomposition


class STLDecomposition(TimeseriesDecomposition):
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
            decomposition = self._DecompositionResults(trend=trend, seasonal=seasonal, residuals=residuals)
        elif self.dku_config.model == "additive":
            self.parameters["endog"] = ts
            stl = STL(**self.parameters)
            statsmodel_results = stl.fit()
            decomposition = self._DecompositionResults()
            decomposition.load(statsmodel_results)
        return decomposition


def format_parameters(dku_config):
    parameters = {"period": dku_config.period}
    if dku_config.advanced:
        parameters["seasonal"] = dku_config.seasonal
        parameters["robust"] = dku_config.robust_stl
        if dku_config.get_param("loess_degrees"):
            parameters["seasonal_deg"] = dku_config.loess_degrees.get("seasonal_deg", 1)
            parameters["trend_deg"] = dku_config.loess_degrees.get("trend_deg", 1)
            parameters["low_pass_deg"] = dku_config.loess_degrees.get("low_pass_deg", 1)
        if dku_config.get_param("speed_jumps"):
            parameters["seasonal_jump"] = dku_config.speed_jumps.get("seasonal_jump", 1)
            parameters["trend_jump"] = dku_config.speed_jumps.get("trend_jump", 1)
            parameters["low_pass_jump"] = dku_config.speed_jumps.get("low_pass_jump", 1)
        if dku_config.get_param("additional_smoothers"):
            parameters["trend"] = dku_config.additional_smoothers.get("trend")
            parameters["low_pass"] = dku_config.additional_smoothers.get("low_pass")
    return parameters
