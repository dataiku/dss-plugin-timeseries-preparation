# -*- coding: utf-8 -*-
import logging

from dataiku.customrecipe import get_recipe_config

from commons import get_resampling_params, get_input_output
from dku_timeseries.resampling.resampling import Resampler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

# --- Setup
(input_dataset, output_dataset) = get_input_output()
recipe_config = get_recipe_config()
datetime_column = recipe_config.get('datetime_column')
if recipe_config.get('advanced_activated') and recipe_config.get('groupby_column'):
    logger.warning("The field 'Column with identifier' is deprecated. Please remove the current value and use the field 'Column with identifiers' instead")
    groupby_columns = [recipe_config.get('groupby_column')]
elif recipe_config.get('advanced_activated') and recipe_config.get('groupby_columns'):
    groupby_columns = recipe_config.get('groupby_columns')
else:
    groupby_columns = None
params = get_resampling_params(recipe_config)

# --- Run
df = input_dataset.get_dataframe()
resampler = Resampler(params)
output_df = resampler.transform(df, datetime_column, groupby_columns=groupby_columns)

# --- Write output
output_dataset.write_with_schema(output_df)
