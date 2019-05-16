# coding: utf-8
from dku_timeseries import ResamplerParams, WindowRoller, WindowRollerParams, SegmentExtractorParams, ExtremaExtractorParams


def get_resampling_params(recipe_config):

	def _p(param_name, default=None):
		return recipe_config.get(param_name, default)

	interpolation_method = _p('interpolation_method')
	extrapolation_method = _p('extrapolation_method')
	time_step_size = float(_p('time_step_size'))
	time_unit = _p('time_unit') 
	offset = int(_p('offset')) #TODO should be float too ?
	crop = int(_p('crop'))

	params = ResamplerParams(interpolation_method = interpolation_method,
			                 extrapolation_method = extrapolation_method,
			                 time_step_size = time_step_size,
			                 time_unit = time_unit,
			                 offset=offset,
			                 crop=crop)

	if _p('advanced_activated'):
		params.groupby_cols = [_p('groupby_col')]

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

	params = WindowRollerParams(window_unit = window_unit,
				                window_width = window_width,
				                window_type = window_type,
				                gaussian_std = gaussian_std,
				                closed_option = closed_option)

	if _p('advanced_activated'):
		params.groupby_cols = [_p('groupby_col')]

	params.check()
	return params

def get_segmenting_params(recipe_config):
	def _p(param_name, default=None):
		return recipe_config.get(param_name, default)
	min_segment_duration_value = _p('min_segment_duration_value')
	max_noise_duration_value = _p('max_noise_duration_value')
	time_unit = _p('time_unit')

	params = SegmentExtractorParams(min_segment_duration_value = min_segment_duration_value,
					                max_noise_duration_value = max_noise_duration_value,
					                time_unit = time_unit)

	if _p('advanced_activated'):
		params.groupby_cols = [_p('groupby_col')]

	params.check()
	return params


def get_extrema_extracting_params(recipe_config):
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

	window_params = WindowRollerParams(window_unit = window_unit,
						                window_width = window_width,
						                window_type = window_type,
						                gaussian_std = gaussian_std,
						                closed_option = closed_option)
	if _p('advanced_activated'):
		window_params.groupby_cols = [_p('groupby_col')]

	window_roller = WindowRoller(window_params)
	params = ExtremaExtractorParams(window_roller = window_roller,
							        extrema_type = extrema_type)

	params.check()
	return params