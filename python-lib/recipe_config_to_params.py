# coding: utf-8
from resampler_params import *

def get_resampling_params(recipe_config):

	def _p(param_name, default=None):
		return recipe_config.get(param_name, default)

	datetime_column = _p('datetime_column')
	interpolation_method = _p('interpolation_method')
	extrapolation_method = _p('extrapolation_method')
	time_step_size = _p('time_step_size')
	time_unit = _p('time_unit')
	offset = _p('offset')
	crop = _p('crop')

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