from statsmodels.tsa.seasonal import seasonal_decompose

from dku_timeseries.dku_decomposition.decomposition import TimeseriesDecomposition, _DecompositionResults


class ClassicalDecomposition(TimeseriesDecomposition):
    """Season-Trend decomposition using moving averages

    Attributes:
        dku_config(DecompositionConfig): mapping structure storing the recipe parameters
        parameters(dict): parameters formatted for the Statsmodel functions

    """

    def __init__(self, dku_config):
        super().__init__(dku_config)
        self.parameters = format_parameters(dku_config)

    def _decompose(self, ts):
        self.parameters["x"] = ts
        statsmodel_results = seasonal_decompose(**self.parameters)
        decomposition = _DecompositionResults()
        decomposition.load(statsmodel_results)
        return decomposition


def format_parameters(dku_config):
    parameters = {"model": dku_config.model, "period": dku_config.season_length}
    if dku_config.advanced and dku_config.get_param("advanced_params"):
        if dku_config.advanced_params.get("extrapolate_trend"):
            parameters["extrapolate_trend"] = dku_config.extrapolate_trend
        if dku_config.advanced_params.get("filt"):
            parameters["filt"] = dku_config.filt
        if dku_config.advanced_params.get("two_sided"):
            parameters["two_sided"] = dku_config.two_sided
    return parameters
