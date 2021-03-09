import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='timeseries-preparation plugin %(levelname)s - %(message)s')


def check_and_get_groupby_columns(recipe_config, dataset_columns):
    long_format = recipe_config.get("advanced_activated", False)
    if long_format:
        groupby_columns = _format_groupby_columns(recipe_config)
        _check_groupby_columns(groupby_columns, dataset_columns)
        return groupby_columns
    else:
        return []


def _format_groupby_columns(recipe_config):
    if recipe_config.get('advanced_activated') and recipe_config.get('groupby_column'):
        logger.warning(
            "The field 'Column with identifier' is deprecated. Please remove the current value and use the field 'Column with identifiers' instead")
        groupby_columns = [recipe_config.get('groupby_column')]
    elif recipe_config.get('advanced_activated') and recipe_config.get('groupby_columns'):
        groupby_columns = recipe_config.get('groupby_columns')
    else:
        groupby_columns = []
    return groupby_columns


def _check_groupby_columns(groupby_columns, dataset_columns):
    if len(groupby_columns) == 0:
        raise ValueError("Long format is activated but no time series identifiers have been provided")
    if not all(identifier in dataset_columns for identifier in groupby_columns):
        raise ValueError("Invalid time series identifiers selection: {}".format(groupby_columns))
