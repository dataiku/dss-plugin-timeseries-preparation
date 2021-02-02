import numpy as np
from statsmodels.tsa.seasonal import STL

from dku_timeseries.decomposition import TimeseriesDecomposition


class STLDecomposition(TimeseriesDecomposition):
    def __init__(self, config):
        super().__init__(config)
        self.parameters = format_parameters(config)

    def _decompose(self, ts):
        if self.config.model_stl == "multiplicative":
            self.parameters["endog"] = np.log(ts)
            stl = STL(**self.parameters)
            results = stl.fit()
            results._trend = np.exp(results.trend)
            results._seasonal = np.exp(results.seasonal)
            results._resid = np.exp(results.resid)
        elif self.config.model_stl == "additive":
            self.parameters["endog"] = ts
            stl = STL(**self.parameters)
            results = stl.fit()
        return results


def format_parameters(config):
    parameters = {"seasonal": config.seasonal}
    if config.advanced:
        parameters["robust"] = config.robust_stl
        if config.loess_degrees:
            parameters["seasonal_deg"] = int(config.loess_degrees.get("seasonal_deg") or 1)
            parameters["trend_deg"] = int(config.loess_degrees.get("trend_deg") or 1)
            parameters["low_pass_deg"] = int(config.loess_degrees.get("low_pass_deg") or 1)
        if config.speed_jumps:
            parameters["seasonal_jump"] = int(config.speed_jumps.get("seasonal_jump") or 1)
            parameters["trend_jump"] = int(config.speed_jumps.get("trend_jump") or 1)
            parameters["low_pass_jump"] = int(config.speed_jumps.get("low_pass_jump") or 1)
    return parameters

