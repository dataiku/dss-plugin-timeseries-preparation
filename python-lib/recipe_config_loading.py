import sys

from dku_config.stl_config import STLConfig
from dku_input_validator.decomposition_input_validator import DecompositionInputValidator
from dku_timeseries import ExtremaExtractorParams
from dku_timeseries import IntervalRestrictorParams
from dku_timeseries import ResamplerParams
from dku_timeseries import WindowAggregator, WindowAggregatorParams
from dku_timeseries.dku_decomposition.stl_decomposition import STLDecomposition
from safe_logger import SafeLogger

logger = SafeLogger("Time series preparation plugin")


def get_resampling_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    interpolation_method = _p('interpolation_method')
    extrapolation_method = _p('extrapolation_method')
    constant_value = _p('constant_value')
    category_imputation_method = _p('category_imputation_method', 'empty')
    category_constant_value = _p('category_constant_value', '')
    time_step = _p('time_step')
    time_unit = _p('time_unit')
    time_unit_end_of_week = _p('time_unit_end_of_week')
    clip_start = _p('clip_start')
    clip_end = _p('clip_end')
    shift = _p('shift')

    params = ResamplerParams(interpolation_method=interpolation_method,
                             extrapolation_method=extrapolation_method,
                             constant_value=constant_value,
                             category_imputation_method=category_imputation_method,
                             category_constant_value=category_constant_value,
                             time_step=time_step,
                             time_unit=time_unit,
                             time_unit_end_of_week=time_unit_end_of_week,
                             clip_start=clip_start,
                             clip_end=clip_end,
                             shift=shift)
    params.check()
    return params


def get_windowing_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    causal_window = _p('causal_window')
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
    aggregation_types = _p('aggregation_types')

    params = WindowAggregatorParams(window_unit=window_unit,
                                    window_width=window_width,
                                    window_type=window_type,
                                    gaussian_std=gaussian_std,
                                    closed_option=closed_option,
                                    causal_window=causal_window,
                                    aggregation_types=aggregation_types)

    params.check()
    return params


def get_interval_restriction_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    min_valid_values_duration_value = _p('min_valid_values_duration_value')
    min_deviation_duration_value = _p('min_deviation_duration_value')
    time_unit = _p('time_unit')

    params = IntervalRestrictorParams(min_valid_values_duration_value=min_valid_values_duration_value,
                                      max_deviation_duration_value=min_deviation_duration_value,
                                      time_unit=time_unit)

    params.check()
    return params


def get_extrema_extraction_params(recipe_config):
    def _p(param_name, default=None):
        return recipe_config.get(param_name, default)

    causal_window = _p('causal_window')
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
    aggregation_types = _p('aggregation_types') + ['retrieve']

    window_params = WindowAggregatorParams(window_unit=window_unit,
                                           window_width=window_width,
                                           window_type=window_type,
                                           gaussian_std=gaussian_std,
                                           closed_option=closed_option,
                                           causal_window=causal_window,
                                           aggregation_types=aggregation_types)

    window_aggregator = WindowAggregator(window_params)
    params = ExtremaExtractorParams(window_aggregator=window_aggregator, extrema_type=extrema_type)
    params.check()
    return params


def get_decomposition_params(config, input_dataset_columns):
    dku_config = STLConfig()
    dku_config.add_parameters(config, input_dataset_columns)
    input_validator = DecompositionInputValidator(dku_config)
    decomposition = STLDecomposition(dku_config)
    return dku_config, input_validator, decomposition


def check_python_version():
    if sys.version_info.major == 2:
        logger.warning(
            "You are using Python {}.{}. Python 2 is now deprecated for the Time Series preparation plugin. Please consider creating a new Python 3.6 "
            "code environment for this plugin".format(sys.version_info.major, sys.version_info.minor))


def check_time_column_parameter(recipe_config, dataset_columns):
    if recipe_config.get("datetime_column") not in dataset_columns:
        raise ValueError("Invalid time column selection: {}".format(recipe_config.get("datetime_column")))


def check_and_get_groupby_columns(recipe_config, dataset_columns):
    long_format = recipe_config.get("advanced_activated", False)
    if long_format:
        groupby_columns = _format_groupby_columns(recipe_config)
        _check_groupby_columns(groupby_columns, dataset_columns)
        return groupby_columns
    else:
        return []


def _format_groupby_columns(recipe_config):
    if recipe_config.get('advanced_activated') and recipe_config.get('groupby_column') and len(recipe_config.get('groupby_columns', [])) == 0:
        logger.warning(
            "The field `Column with identifier` is deprecated. It is now replaced with the field `Time series identifiers`, which allows for several "
            "identifiers. That is why you should preferably use the field 'Time series identifiers'. You can still use 'Column with identifier' if you "
            "have one identifier only")
        groupby_columns = [recipe_config.get('groupby_column')]
    elif recipe_config.get('advanced_activated') and recipe_config.get('groupby_columns'):
        if recipe_config.get('groupby_column'):
            logger.warning("The fields `Column with identifier` and `Time series identifiers` both contain a value. As `Column with identifiers`is deprecated, "
                           "the recipe will only consider the value of `Time series identifiers`. ")
        groupby_columns = recipe_config.get('groupby_columns')
    else:
        groupby_columns = []
    return groupby_columns


def _check_groupby_columns(groupby_columns, dataset_columns):
    if len(groupby_columns) == 0:
        raise ValueError("Long format is activated but no time series identifiers have been provided")
    if not all(identifier in dataset_columns for identifier in groupby_columns):
        raise ValueError("Invalid time series identifiers selection: {}".format(groupby_columns))
