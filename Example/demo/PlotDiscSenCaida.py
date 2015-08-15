from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")

from SBDet import *
import pylab as P


def w2():
    tr = zload('./w2_influe_res-2.pkz')

    w2_set = tr['w2_set']
    stat = tr['stat']
    tp, fn, tn, fp, fpr, tpr = zip(*stat)
    P.plot(w2_set, fp)
    P.plot(w2_set, tp)
    P.plot(w2_set, np.array(tp) - np.array(fp))
    P.show()

# w2()

tr = zload('./viz-com-test.pkz')
A = tr['A']
solution = tr['solution']
botnet_nodes = tr['botnet_nodes']
mix_nodes = tr['mix_nodes']
nc = ['blue', 'red', 'green', 'black', 'm']
ns = 'os><x'
pos = zload('graph_pos.pkz')
# import ipdb;ipdb.set_trace()
shrink_map = tr['shrink_map']


import igraph
ig = igraph.Graph(directed=False)
ig.add_vertices(range(len(shrink_map)))
I, J = A.nonzero()
edges = zip(I, J)
ig.add_edges(edges)
# res = ig.community_leading_eigenvector(clusters=2)
# res = ig.community_leading_eigenvector(clusters=-1)
res = ig.community_leading_eigenvector(clusters=-1)
solution_leading_eigen = res.membership
# import ipdb;ipdb.set_trace()



res = ig.community_infomap(trials=50)
solution_info_map = res.membership


solution_walk_trap = ig.community_walktrap().as_clustering(3).membership

tr['solution_walk_trap'] = solution_walk_trap
tr['solution_leading_eigen'] = solution_leading_eigen


# P.figure((800,600))
P.figure()
# P.show()
# import sys; sys.exit(0)
P.subplot(221)
        # pos='graphviz',
        # pos='spring',
pos = draw_graph(A,
        None,
        pic_show=False,
        pos=pos,
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i > 0)] for i in solution],
        node_shape=[ns[int(i > 0)] for i in solution],
        edge_color='grey',
        # pic_name='./det_res_sdp.pdf'
        )
P.title('SBDet')
        # pos=pos,


P.subplot(222)
# P.figure()
pos = draw_graph(A,
        None,
        pic_show=False,
        # pos=pos,
        pos='graphviz',
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i)] for i in solution_leading_eigen],
        node_shape=[ns[int(i)] for i in solution_leading_eigen],
        edge_color='grey',
        )
P.title('LeadingEigen')



P.subplot(223)
# P.figure()
pos = draw_graph(A,
        None,
        pic_show=False,
        pos=pos,
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i > 0)] for i in tr['ref_sol']],
        node_shape=[ns[int(i > 0)] for i in tr['ref_sol']],
        )
P.title('GroundTruth')


P.subplot(224)
pos = draw_graph(A,
        None,
        pic_show=False,
        pos=pos,
        with_labels=False,
        node_size=100,
        node_color=[nc[int(i > 0)] for i in solution_info_map],
        node_shape=[ns[int(i > 0)] for i in solution_info_map],
        edge_color='grey',
        )
P.title('InfoMap')

P.show()