# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='timeseries-preparation plugin %(levelname)s - %(message)s')


TIME_STEP_MAPPING = {
    'year': 'A',
    'quarter': 'Q',
    'month': 'M',
    'week': 'W',
    'day': 'D',
    'hour': 'H',
    'minute': 'T',
    'second': 'S',
    'millisecond': 'L'
}

OFFSET_MAPPING = {
    'day': 'D',
    'hour': 'h',
    'minute': 'm',
    'second': 's',
    'millisecond': 'ms'
}

INTERPOLATION_METHODS = ['None', 'nearest', 'previous',
                         'next', 'linear', 'quadratic', 'cubic', 'barycentric']
EXTRAPOLATION_METHODS = ['None', 'clip', 'interpolation']
TIME_UNITS = ['year', 'quarter', 'month', 'week',
              'day', 'hour', 'minute', 'second', 'millisecond', 'row']
OFFSET_UNITS = ['day', 'hour', 'minute', 'second', 'millisecond', 'row']


class ResamplerParams:

    def __init__(self,
                 interpolation_method='linear',
                 extrapolation_method='clip',
                 time_step_size=1,
                 time_unit='second',
                 offset=0,
                 crop=0,
                 groupby_cols=[]):

        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.time_step_size = time_step_size
        self.time_unit = time_unit
        self.resampling_step = str(
            self.time_step_size) + TIME_STEP_MAPPING.get(self.time_unit, '')
        self.offset = offset
        self.offset_step = str(self.offset) + \
            OFFSET_MAPPING.get(self.time_unit, '')
        self.crop = crop
        self.crop_step = str(self.crop) + \
            OFFSET_MAPPING.get(self.time_unit, '')
        self.groupby_cols = groupby_cols

    def check(self):

        if self.interpolation_method not in INTERPOLATION_METHODS:
            raise ValueError('Method "{0}" is not valid. Possible interpolation methods are: {1}.'.format(
                self.interpolation_method, INTERPOLATION_METHODS))

        if self.extrapolation_method not in EXTRAPOLATION_METHODS:
            raise ValueError('Method "{0}" is not valid. Possible extrapolation methods are: {1}.'.format(
                self.extrapolation_method, EXTRAPOLATION_METHODS))

        if self.time_step_size < 0:
            raise ValueError('Time step can not be negative.')

        if self.time_unit not in TIME_UNITS:
            raise ValueError('"{0}" is not a valid unit. Possible time units are: {1}'.format(
                self.time_unit, TIME_UNITS))

        if self.offset != 0 and self.time_unit not in OFFSET_UNITS:
            raise ValueError('Can not use offset with "{0}" unit. Possible time units are: {1}'.format(
                self.time_unit, OFFSET_UNITS))
        if self.crop != 0 and self.time_unit not in OFFSET_UNITS:
            raise ValueError('Can not use crop with "{0}" unit. Possible time units are: {1}'.format(
                self.time_unit, OFFSET_UNITS))
