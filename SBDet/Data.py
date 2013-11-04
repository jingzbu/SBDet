"""  Defines all the wrapper classes for a variety of `data types`
    There are two categrories:
        1. Files in hard disk. the base class is the :class:`PreloadHardDiskFile`.
        2. MySQL database. The base class is the :class:`MySQLDatabase`.
"""
from __future__ import print_function, division, absolute_import
from .Util import abstract_method
from .Util import Find, DataEndException
# from Util import np
import numpy as np


class Data(object):
    """abstract base class for data. Data class deals with any implementation
    details of the data. it can be a file, a sql data base, and so on, as long
    as it supports the pure virtual methods defined here.
    """
    def get_rows(self, fields=None, rg=None, rg_type=None):
        """ get a slice of feature

        Parameters
        ---------------
        fields : string or list of string
            the fields we need to get
        rg : list of two floats
            is the range for the slice
        rg_type : str,  {'flow', 'time'}
            type for range

        Returns
        --------------
        list of list

        """
        abstract_method()

    def get_where(self, rg=None, rg_type=None):
        """ get the absolute position of flows records that within the range.

        Find all flows such that its belong to [rg[0], rg[1]). The interval
        is closed in the starting point and open in the ending pont.

        Parameters
        ------------------
        rg : list or tuple or None
            range of the the data. If rg == None, simply return position
            (0, row_num])
        rg_type : {'flow', 'time'}
            specify the type of the range.

        Returns
        -------------------
        sp, ep : ints
            flows with index such that sp <= idx < ed belongs to the range

        """
        abstract_method()

    def get_min_max(self, field_list):
        """  get the min and max value for fields

        Parameters
        ---------------
        field_list : a list of str

        Returns
        --------------
        miN, maX : a list of floats
            the mimium(maximium) value for each field in field_list
        """
        abstract_method()

import re


def parse_records(f_name, FORMAT, regular_expression):
    flow = []
    with open(f_name, 'r') as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            if line == '\n':  # Ignore Blank Line
                continue
            item = re.split(regular_expression, line)
            f = tuple(h(item[pos]) for k, pos, h in FORMAT)
            flow.append(f)
    return flow

IP = lambda x:tuple(np.uint8(v) for v in x.rsplit('.'))


class PreloadHardDiskFile(Data):
    """ abstract base class for hard disk file The flow file into memory as a
    whole, so it cannot deal with flow file larger than your memery
    """

    RE = None
    """regular expression used to seperate each line into segments"""

    FORMAT = None
    """Format of the Data. Should be a list of tuple, each tuple has 3
    element (field_name, position, converter).
    """

    DT = None
    """ Specify how the data will be stored in Numpy array. Should be np.dtype
    See
    `Numpy.dtype
    <http://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html>`_
    for more information.
    """

    @property
    def FIELDS(self):
        return zip(*self.FORMAT)[0] if self.FORMAT is not None else None

    """  The name of all columns """

    def __init__(self, f_name):
        """ data_order can be flow_first | feature_first
        """
        self.f_name = f_name
        self._init()

    @staticmethod
    def parse(*argv, **kwargv):
        return parse_records(*argv, **kwargv)

    def _init(self):
        self.fea_vec = self.parse(self.f_name, self.FORMAT, self.RE)
        self.table = np.array(self.fea_vec, dtype=self.DT)
        self.row_num = self.table.shape[0]

        self.t = np.array([t for t in self.get_rows('start_time')])
        t_idx = np.argsort(self.t)
        self.table = self.table[t_idx]
        self.t = self.t[t_idx]

        self.min_time = min(self.t)
        self.max_time = max(self.t)

    def get_where(self, rg=None, rg_type=None):
        if not rg:
            return 0, self.row_num
        if rg_type == 'flow':
            sp, ep = rg
            if sp >= self.row_num:
                raise DataEndException()
        elif rg_type == 'time':
            sp = Find(self.t, rg[0] + self.min_time)
            ep = Find(self.t, rg[1] + self.min_time)
            assert(sp != -1 and ep != -1)
            if (sp == len(self.t) - 1 or ep == len(self.t) - 1):
                raise DataEndException()
        else:
            raise ValueError('unknow window type')
        return sp, ep

    def get_rows(self, fields=None, rg=None, rg_type=None, row_indices=None):
        if fields is None:
            fields = list(self.FIELDS)

        if row_indices is not None:
            return self.table[row_indices][fields]

        sp, ep = self.get_where(rg, rg_type)
        return self.table[sp:ep][fields]

    def get_min_max(self, feas):
        min_vec = []
        max_vec = []
        for fea in feas:
            dat = self.get_rows(fea)
            min_vec.append(min(dat))
            max_vec.append(max(dat))
        return min_vec, max_vec

import datetime
import time


def str_to_sec(ss, formats):
    """
    >>> str_to_sec('2012-06-17T16:26:18.300868', '%Y-%m-%dT%H:%M:%S.%f')
    14660778.300868
    """
    # x = time.strptime(ss,'%Y-%m-%dT%H:%M:%S.%f')
    x = time.strptime(ss,formats)

    ts = ss.rsplit('.')[1]
    micros = int(ts) if len(ts) == 6 else 0  # FIXME Add microseconds support for xflow
    return datetime.timedelta(
        days=x.tm_yday,
        hours=x.tm_hour,
        minutes=x.tm_min,
        seconds=x.tm_sec,
        microseconds=micros,
    ).total_seconds()


class HDF_FlowExporter(PreloadHardDiskFile):
    """  Data generated FlowExporter. It is a simple tool to convert pcap to
    flow data. It is avaliable in tools folder.

    """
    RE = '[ \n]'
    FORMAT = [
        ('start_time', 0, np.float64),
        ('src_ip', 1, IP),
        ('dst_ip', 2, IP),
        ('prot', 3, np.str_),
        ('flow_size', 4, np.float64),
        ('duration', 5, np.float64),
    ]
    DT = np.dtype([
        ('start_time', np.float64, 1),
        ('src_ip', np.uint8, (4,)),
        ('dst_ip', np.uint8, (4,)),
        ('prot', np.str_, 5),
        ('flow_size', np.float64, 1),
        ('duration', np.float64, 1),
    ])


def seq_convert(args, arg_num, handlers):
    res = []
    i = 0
    for n, h in zip(arg_num, handlers):
        res.append(h(*args[i:(i + n)]))
        i += n
    return tuple(res)

if __name__ == "__main__":
    import doctest
    doctest.testmod()