# coding: utf-8
import numpy as np

def has_duplicates(df, column):
    return any(df.duplicated(subset=[column], keep=False))


def nothing_to_do(df, min_len=2):
    return len(df) < min_len


def filter_empty_columns(df, columns):
    filtered_columns = []
    for col in columns:
        if np.sum(df[col].notnull()) > 0:
            filtered_columns.append(col)
    return filtered_columns


def generic_check_compute_arguments(datetime_column, groupby_columns):
    if not isinstance(datetime_column, basestring):
        raise ValueError('datetime_column param must be string. Got: ' + str(datetime_column))
    if groupby_columns:
        if not isinstance(groupby_columns, list):
            raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(groupby_columns))
        for col in groupby_columns:
            if not isinstance(col, basestring):
                raise ValueError('groupby_columns param must be an array of strings. Got: ' + str(col))