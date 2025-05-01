# coding: utf-8
import logging

import dataiku
from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')


def get_input_output():
    if len(get_input_names_for_role('input_dataset')) == 0:
        raise ValueError('No input dataset.')
    input_dataset_name = get_input_names_for_role('input_dataset')[0]
    input_dataset = dataiku.Dataset(input_dataset_name)
    if len(get_output_names_for_role('output_dataset')) == 0:
        raise ValueError('No output dataset.')
    output_dataset_name = get_output_names_for_role('output_dataset')[0]
    output_dataset = dataiku.Dataset(output_dataset_name)
    return (input_dataset, output_dataset)


def set_column_description(output_dataset, column_description_dict, input_dataset):
    """
    Set column descriptions of the output dataset based on a dictionary of column descriptions
    and retains the column descriptions from the input dataset (optional) if the column name matches.
    """
    if input_dataset is None:
        input_dataset_schema = []
    else:
        input_dataset_schema = input_dataset.read_schema()
    output_dataset_schema = output_dataset.read_schema()
    input_columns_names = [col["name"] for col in input_dataset_schema]
    for output_col_info in output_dataset_schema:
        output_col_name = output_col_info.get("name", "")
        output_col_info["comment"] = column_description_dict.get(output_col_name)
        if output_col_name in input_columns_names:
            matched_comment = [
                input_col_info.get("comment", "")
                for input_col_info in input_dataset_schema
                if input_col_info.get("name") == output_col_name
            ]
            if len(matched_comment) != 0:
                output_col_info["comment"] = matched_comment[0]
    output_dataset.write_schema(output_dataset_schema)
