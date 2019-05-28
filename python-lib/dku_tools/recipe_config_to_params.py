# coding: utf-8
from dku_timeseries import ResamplerParams, WindowAggregator, WindowAggregatorParams, SegmentExtractorParams, \
    ExtremaExtractorParams


def get_resampling_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    interpolation_method = _p('interpolation_method')
    extrapolation_method = _p('extrapolation_method')
    time_step = float(_p('time_step'))
    time_unit = _p('time_unit')
    clip_start = int(_p('clip_start'))  # TODO should be float too ?
    clip_end = int(_p('clip_end'))

    params = ResamplerParams(interpolation_method=interpolation_method,
                             extrapolation_method=extrapolation_method,
                             time_step=time_step,
                             time_unit=time_unit,
                             clip_start=clip_start,
                             clip_end=clip_end)
    params.check()
    return params


def get_windowing_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    window_unit = _p('window_unit')
    window_width = int(_p('window_width'))
    if _p('window_type') == 'none':
        window_type = None
    else:
        window_type = _p('window_type')

    if window_type == 'gaussian':
        gaussian_std = _p('gaussian_std')
    else:
        gaussian_std = None

    closed_option = _p('closed_option')

    params = WindowAggregatorParams(window_unit=window_unit,
                                window_width=window_width,
                                window_type=window_type,
                                gaussian_std=gaussian_std,
                                closed_option=closed_option)

    params.check()
    return params


def get_segmenting_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    min_segment_duration_value = _p('min_segment_duration_value')
    max_noise_duration_value = _p('max_noise_duration_value')
    time_unit = _p('time_unit')

    params = SegmentExtractorParams(min_segment_duration_value=min_segment_duration_value,
                                    max_noise_duration_value=max_noise_duration_value,
                                    time_unit=time_unit)

    params.check()
    return params


def get_extrema_extraction_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    window_unit = _p('window_unit')
    window_width = int(_p('window_width'))
    if _p('window_type') == 'none':
        window_type = None
    else:
        window_type = _p('window_type')

    if window_type == 'gaussian':
        gaussian_std = _p('gaussian_std')
    else:
        gaussian_std = None

    closed_option = _p('closed_option')

    extrema_type = _p('extrema_type')

    window_params = WindowAggregatorParams(window_unit=window_unit,
                                       window_width=window_width,
                                       window_type=window_type,
                                       gaussian_std=gaussian_std,
                                       closed_option=closed_option)

    window_aggregator = WindowAggregator(window_params)
    params = ExtremaExtractorParams(window_aggregator=window_aggregator,
                                    extrema_type=extrema_type)

    params.check()
    return params
