# -*- coding: utf-8 -*-
import logging

from dataiku.customrecipe import get_recipe_config

from commons import get_resampling_params, get_input_output
from recipe_config_loading import check_and_get_groupby_columns
from dku_timeseries import Resampler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')

# --- Setup
(input_dataset, output_dataset) = get_input_output()
recipe_config = get_recipe_config()
input_dataset_columns = [column["name"] for column in input_dataset.read_schema()]
groupby_columns = check_and_get_groupby_columns(recipe_config, input_dataset_columns)
datetime_column = recipe_config.get('datetime_column')
params = get_resampling_params(recipe_config)

# --- Run
df = input_dataset.get_dataframe()
resampler = Resampler(params)
output_df = resampler.transform(df, datetime_column, groupby_columns=groupby_columns)

# --- Write output
output_dataset.write_with_schema(output_df)
