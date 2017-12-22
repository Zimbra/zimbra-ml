"""
(C) 2017, Michael Toutonghi - All rights reserved.

Author: Michael Toutonghi
Creation date: 5/7/2017

Licensed to Synacor for non-exclusive, unlimited use, reproduction, and derivation.

This file includes classes to fill in the blanks around available libraries and make some efficient
but important data operations easier.

Includes:
DateTimeFormats: Easy manipulation of dates and times, with simple timezone support for data retrieval and storage
ArrayFields: Access an array of values through python fields, gaining the best of both worlds when dealing with large
amounts of data


"""
import pytz
import datetime
import pandas as pd
import numpy as np
from collections import Sequence

class DateTimeFormats():
    @staticmethod
    def timestamp_to_naive_datetime(ts):
        return pd.datetime.fromtimestamp(ts)

    @staticmethod
    def naive_local_to_naive_utc(dt, local_zone):
        if isinstance(local_zone, str):
            local_zone = pytz.timezone(local_zone)
        return local_zone.localize(dt).astimezone(pytz.utc).replace(tzinfo=None)

    @staticmethod
    def naive_utc_to_naive_local(dt, local_zone):
        if isinstance(local_zone, str):
            local_zone = pytz.timezone(local_zone)
        return pytz.utc.localize(dt).astimezone(local_zone).replace(tzinfo=None)

    @staticmethod
    def datetime_as_datastring(dt):
        assert isinstance(dt, datetime.datetime)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

class FieldIndex(object):
    def __init__(self, field_list):
        """
        This basically just precompiles a set of indexes into an object that can provide faster access to
        array indexes through named fields
        :param field_dict:
        """
        for k, i in zip(field_list, range(len(field_list))):
            setattr(self, k, i)

    def __getitem__(self, item):
        return getattr(self, item, None)


class ArrayFields(Sequence):
    class Property(object):
        def __init__(self, index):
            self.i = index

        def _prop_get(self, other):
            return other.array[self.i]

        def _prop_set(self, other, value):
            other.array[self.i] = value

    def __init__(self, array, field_list, offset=0, columns=None):
        """
        enables field like access to an array, as well as sub-indexing on a set of
        fields by number within an array. the elements can also be iterated by value,
        and the property 'field_list' allows iteration by field names.
        :param array:
        :param field_index:
        """
        self.array = array
        self._offset = offset
        self._fields = field_list
        cls = type(self)
        if not getattr(cls, '__perinstance', False):
            cls = type(cls.__name__, (cls,), {})
            cls.__perinstance = True
            self.__class__ = cls

        for k, i in zip(field_list, range(len(field_list))):
            sl = np.s_[offset + i] if len (array.shape) == 1 or columns is None else np.s_[offset + i, columns]
            prop = ArrayFields.Property(sl)
            setattr(cls, k, property(prop._prop_get, prop._prop_set))

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index >= len(self):
                raise IndexError
            else:
                return self.array[index + self._offset]
        else:
            return getattr(self, index)

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if index >= len(self):
                raise IndexError
            else:
                self.array[index + self._offset] = value
        else:
            return setattr(self, index, value)
