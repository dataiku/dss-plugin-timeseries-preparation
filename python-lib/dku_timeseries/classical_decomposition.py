from statsmodels.tsa.seasonal import seasonal_decompose

from dku_timeseries.decomposition import TimeseriesDecomposition


class ClassicalDecomposition(TimeseriesDecomposition):
    def __init__(self, config):
        super().__init__(config)
        self.parameters = format_parameters(config)

    def _decompose(self, ts):
        self.parameters["x"] = ts
        results = seasonal_decompose(**self.parameters)
        return results


def format_parameters(config):
    parameters = {"model": config.model}
    if config.advanced:
        if config.advanced_params.get("extrapolate_trend"):
            if config.extrapolate_trend != "freq":
                parameters["extrapolate_trend"] = int(config.extrapolate_trend)
            else:
                parameters["extrapolate_trend"] = "freq"

        if config.advanced_params.get("filt"):
            parameters["filt"] = config.filt

        if config.advanced_params.get("two_sided"):
            if config.two_sided == "True":
                parameters["two_sided"] = True
            elif config.two_sided == "False":
                parameters["two_sided"] = False

    return parameters
