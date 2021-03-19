# -*- coding: utf-8 -*-
import logging
from dataiku.customrecipe import get_recipe_config
from commons import *
from recipe_config_loading import check_time_column_parameter, check_and_get_groupby_columns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

# --- Setup
(input_dataset, output_dataset) = get_input_output()
recipe_config = get_recipe_config()
input_dataset_columns = [column["name"] for column in input_dataset.read_schema()]
check_time_column_parameter(recipe_config, input_dataset_columns)
datetime_column = recipe_config.get('datetime_column')
groupby_columns = check_and_get_groupby_columns(recipe_config, input_dataset_columns)
params = get_windowing_params(recipe_config)

# --- Run
df = input_dataset.get_dataframe()
window_aggregator = WindowAggregator(params)
output_df = window_aggregator.compute(df, datetime_column, groupby_columns=groupby_columns)

# --- Write output
output_dataset.write_with_schema(output_df)
