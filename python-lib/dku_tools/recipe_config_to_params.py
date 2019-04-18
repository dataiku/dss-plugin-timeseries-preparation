# coding: utf-8
from dku_timeseries import ResamplerParams, WindowRollerParams

def get_resampling_params(recipe_config):

	def _p(param_name, default=None):
		return recipe_config.get(param_name, default)

	datetime_column = _p('datetime_column')
	interpolation_method = _p('interpolation_method')
	extrapolation_method = _p('extrapolation_method')
	time_step_size = int(_p('time_step_size'))
	time_unit = _p('time_unit')
	offset = int(_p('offset'))
	crop = int(_p('crop'))

	params = ResamplerParams(datetime_column = datetime_column, 
							 interpolation_method = interpolation_method,
			                 extrapolation_method = extrapolation_method,
			                 time_step_size = time_step_size,
			                 time_unit = time_unit,
			                 offset=offset,
			                 crop=crop)

	if _p('advance_activate'):
		params.groupby_cols = [_p('groupby_cols')]

	params.check()
	return params

def get_windowing_params(recipe_config):

	def _p(param_name, default=None):
		return recipe_config.get(param_name, default)

	datetime_column = _p('datetime_column')
	window_unit = _p('window_unit')
	window_width = int(_p('window_width'))
	if _p('window_type') == 'none':
		window_type = None
	else:
		window_type = _p('window_type')

	if window_type == 'gaussian':
		gaussian_std = _p(gaussian_std)
	else:
		gaussian_std = None

	closed_option = _p('closed_option')

	params = WindowRollerParams(datetime_column = datetime_column, 
								 window_unit = window_unit,
				                 window_width = window_width,
				                 window_type = window_type,
				                 gaussian_std = gaussian_std,
				                 closed_option = closed_option
				                )

	if _p('advance_activate'):
		params.groupby_cols = [_p('groupby_cols')]

	params.check()
	return params