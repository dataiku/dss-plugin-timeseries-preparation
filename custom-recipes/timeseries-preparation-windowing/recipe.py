# -*- coding: utf-8 -*-
import logging

from dataiku.customrecipe import get_recipe_config

from commons import check_python_version, get_input_output, get_windowing_params
from dku_timeseries import WindowAggregator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

check_python_version()

# --- Setup
(input_dataset, output_dataset) = get_input_output()
recipe_config = get_recipe_config()
datetime_column = recipe_config.get('datetime_column')
if recipe_config.get('advanced_activated') and recipe_config.get('groupby_column'):
    groupby_columns = [recipe_config.get('groupby_column')]
else:
    groupby_columns = None
params = get_windowing_params(recipe_config)

# --- Run
df = input_dataset.get_dataframe()
window_aggregator = WindowAggregator(params)
output_df = window_aggregator.compute(df, datetime_column, groupby_columns=groupby_columns)

# --- Write output
output_dataset.write_with_schema(output_df)
