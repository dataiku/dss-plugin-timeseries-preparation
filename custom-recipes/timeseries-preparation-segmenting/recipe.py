# -*- coding: utf-8 -*-
import dataiku
from dataiku.customrecipe import *
import logging
from dku_timeseries import SegmentExtractor
from dku_tools import get_segmenting_params

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,  
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')


# --- Get IOs
try:
	input_dataset_name = get_input_names_for_role('input_dataset')[0]
except:
	raise ValueError('No input dataset.') 
input_dataset = dataiku.Dataset(input_dataset_name)

output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

# --- Get configuration
recipe_config = get_recipe_config()
datetime_column = recipe_config.get('datetime_column')
segment_column = recipe_config.get('segment_column')
min_threshold = recipe_config.get('min_threshold')
max_threshold = recipe_config.get('max_threshold')
threshold_dict = {segment_column: (min_threshold, max_threshold)}
params = get_segmenting_params(recipe_config)

# --- Run
df = input_dataset.get_dataframe()
segment_extractor = SegmentExtractor(params)
output_df = segment_extractor.compute(df, datetime_column, threshold_dict)

# --- Write output
output_dataset.write_with_schema(output_df)
