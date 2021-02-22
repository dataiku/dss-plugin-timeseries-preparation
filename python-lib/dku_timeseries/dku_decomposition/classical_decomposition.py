from statsmodels.tsa.seasonal import seasonal_decompose

from dku_timeseries.dku_decomposition.decomposition import TimeseriesDecomposition


class ClassicalDecomposition(TimeseriesDecomposition):
    def __init__(self, dku_config):
        super().__init__(dku_config)
        self.parameters = format_parameters(dku_config)

    def _decompose(self, ts):
        self.parameters["x"] = ts
        statsmodel_results = seasonal_decompose(**self.parameters)
        decomposition = self._DecompositionResults()
        decomposition.load(statsmodel_results)
        return decomposition


def format_parameters(dku_config):
    parameters = {"model": dku_config.model, "period": dku_config.period}
    if dku_config.advanced:
        if dku_config.advanced_params.get("extrapolate_trend"):
            parameters["extrapolate_trend"] = dku_config.extrapolate_trend
        if dku_config.advanced_params.get("filt"):
            parameters["filt"] = dku_config.filt
        if dku_config.advanced_params.get("two_sided"):
            parameters["two_sided"] = dku_config.two_sided
    return parameters
