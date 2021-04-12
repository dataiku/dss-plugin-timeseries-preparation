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
