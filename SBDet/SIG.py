#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import os
import sys
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import Counter

from .Data import HDF_FlowExporter
# from Dataset import HDF_FlowExporter
from .CGraph import Igraph
from .CGraph import NetworkXGraph
from .Util import dump, load
# from .Util import I1
# from .Util import np_index, np_to_dotted
from .Util import progress_bar
from .Util import DataEndException
from .Util import igraph
# from .Util import sp, la, stats


def cal_SIG_low_mem(data_file, interval=10.0, dur=10.0,
                    rg=(0.0, float('inf')), folder=None):
    """ Calculate the Social Interaction Graph (SIG)

    Parameters
    ---------------
    data_file : Data class
        flows data file
    interval, dur : float
        interval between two windows and the duration of each window
    rg : tuple (start_time, end_time)
        Only flows whose timestamps belong to [start_time, end_time) are used.
    folder : str
        path for the output folder
    """
    start_time, end_time = rg
    N = sys.maxint if end_time == float('inf') \
        else int((end_time - start_time) // dur)

    ips = NetworkXGraph(data_file).get_vertices()
    dump(ips, folder + 'nodes.pk')
    try:
        for i in xrange(N):
            sys.stdout.write("\r[%d]" % i)
            sys.stdout.flush()

            tg = NetworkXGraph(data_file)
            tg.add_vertices(ips)
            seg = [start_time + i * interval, start_time + i * interval + dur]
            records = tg.filter(prot=None, rg=seg, rg_type='time')
            edges = tg.get_edges(records)
            tg.add_edges(edges)
            dump({'edges': edges}, '%s%i.pk' % (folder, i))
    except DataEndException:
        print('reach end')

    sys.stdout.write('\n')


def pack_sigs(folder, out):
    """  pack the sequence of sigs in a **folder** into a single file
    """
    N = sys.maxint
    try:
        nodes = load('%snodes.pk' % (folder))
        nm = dict(zip(nodes, range(len(nodes))))
        sigs = []
        for i in xrange(N):
            if i % 100 == 0:
                sys.stdout.write("\r%d" % i)
                sys.stdout.flush()
            dat = load('%s%i.pk' % (folder, i))
            ce = lambda edge: (nm[edge[0]], nm[edge[1]])
            sigs.append([(ce(e), c) for e, c in dat['edges'].iteritems()])
    except IOError:
        pass
    dump({'nodes': nodes, 'sig_edges': sigs}, out)
    sys.stdout.write('\n')


def to_sigs(data, out_folder, dur):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    cal_SIG_low_mem(data,
                    interval=dur,
                    dur=dur,
                    folder=out_folder)
    pack_sigs(out_folder, out_folder+'sigs.pk')


def cal_SIG(data_file, interval=10.0, dur=10.0, rg=(0.0, float('inf')),
            directed=False, tp='igraph'):
    """ Calculate the Social Interaction Graph (SIG)

    Parameters
    ---------------
    data_file : str
        path of the flows data file
    interval, dur : float
        interval between two windows and the duration of each window
    rg : tuple (start_time, end_time)
        Only flows whose timestamps belong to [start_time, end_time) are used.
    directed : bool
        if true, the SIG is directed, otherwise it is undirected.
    tp : {'igraph', 'networkx'}
        type of graph

    Returns
    --------------
    sigs : list of Graphs with format specified by tp

    """
    start_time, end_time = rg
    if end_time == float('inf'):
        N = sys.maxint
    else:
        N = int((end_time - start_time) // dur)

    sigs = []
    if isinstance(data_file, str):
        data = HDF_FlowExporter(data_file)
    else:
        data = data_file

    # TGraph = NetworkXGraph
    TGraph_map = {
        'igraph': Igraph,
        'networkx': NetworkXGraph,
    }
    TGraph = TGraph_map[tp]

    ips = TGraph(data).get_vertices()
    # ips = [np_to_dotted(ip) for ip in ips]
    try:
        for i in xrange(N):
            progress_bar(i * 1.0 / N * 100)
            tg = TGraph(data)
            tg.add_vertices(ips)
            records = tg.filter(prot=None,
                                rg=[start_time + i * interval,
                                    start_time + i * interval + dur],
                                rg_type='time')
            edges = tg.get_edges(records)
            tg.add_edges(edges)
            sigs.append(tg.graph)
    except DataEndException:
        print('reach end')

    return sigs


def animate_SIGs(sigs, ani_folder):
    if not isinstance(sigs[0], igraph.Graph):
        raise Exception("animate_SIGs only works with python-igraph")
    layout = None
    if not os.path.exists(ani_folder):
        os.mkdir(ani_folder)

    N = len(sigs)
    print('animation progress:')
    for i, ig in enumerate(sigs):
        progress_bar(i * 1.0 / N * 100)
        # nig = NetworkXGraph(graph=ig)
        nig = Igraph(graph=ig)
        layout = nig.gen_layout() if layout is None else layout
        nig.plot(ani_folder + "%04d.png" % (i), layout=layout)