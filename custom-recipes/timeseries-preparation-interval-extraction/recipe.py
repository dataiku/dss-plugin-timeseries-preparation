# -*- coding: utf-8 -*-
import logging
from dataiku.customrecipe import *
from dku_timeseries import IntervalRestrictor
from commons import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

# --- Setup
(input_dataset, output_dataset) = get_input_output()
recipe_config = get_recipe_config()
datetime_column = recipe_config.get('datetime_column')
value_column = recipe_config.get('value_column')
min_threshold = recipe_config.get('min_threshold')
max_threshold = recipe_config.get('max_threshold')
threshold_dict = {value_column: (min_threshold, max_threshold)}
if recipe_config.get('advanced_activated') and recipe_config.get('groupby_column'):
    groupby_columns = [recipe_config.get('groupby_column')]
else:
    groupby_columns = None
params = get_interval_restriction_params(recipe_config)

# --- Run
df = input_dataset.get_dataframe()
interval_restrictor = IntervalRestrictor(params)
output_df = interval_restrictor.compute(df, datetime_column, threshold_dict, groupby_columns=groupby_columns)

# --- Write output
output_dataset.write_with_schema(output_df)
