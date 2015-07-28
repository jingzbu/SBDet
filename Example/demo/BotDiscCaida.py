from __future__ import print_function, division, absolute_import
import sys
sys.path.insert(0, "../../")
from SBDet import *
import pylab as P
from subprocess import check_call
import networkx as nx

###############################################################
# Step 1. Generate Dataset by mixing background and botnet SIGs
###############################################################

# Read background SIGs
bg_sigs, bg_nodes = parseToCoo('capture20110816.binetflow.slice_13_1.sigs',
        undirected=True)

bg_sigs = bg_sigs[0:2000]

deg_samples = mg_sample(len(bg_nodes), bg_sigs, 100, 50)
ER_para, ER_lk = mle(deg_samples, 'ER')

print(ER_para, ER_lk)



def symmetrize(a):
        return a + a.T

def convert(sigs):
    return [symmetrize(g.tocsr()) for g in sigs]



# Read botnet SIGs
botnet_sigs, botnet_nodes = parseToCoo('ddostrace.sigs',
                                       undirected=True)

mix_sigs, mix_nodes = mix_append((bg_sigs, bg_nodes),
        (botnet_sigs, botnet_nodes), 200)




###############################################################
# Step 2. Anomaly Detection
###############################################################
divs = monitor_deg_dis(mix_sigs, 'ER', (ER_para, 1e-10), minlength=None)



THRE = 0.01

# The index of suspicious SIGs
det_idx = [i for i, div in enumerate(divs) if div > THRE]

bot_adjs = [mix_sigs[idx] for idx in det_idx]

# convert bot_adjs to symmetric and to csr format
bot_adjs = convert(bot_adjs)

print(bot_adjs)

# assert(1==2)

###############################################################
# Step 3. Botnet Discovery
###############################################################

pivot_th = 0.01

cor_th = 0.005

w1 = 2
w2 = 0.001

lamb = 0

# botnet = detect_botnet(bot_adjs, pivot_th, cor_th, w1, w2, lamb)
node_num = bot_adjs[0].shape[0]
weights = np.ones((node_num, )) / node_num  # equal weights
sigs = bot_adjs
p_nodes, total_inta_mat = ident_pivot_nodes(bot_adjs, weights, pivot_th)
print('p_nodes', p_nodes)



inta = cal_inta_pnodes(sigs, weights, p_nodes, total_inta_mat[p_nodes])

A, npcor = cal_cor_graph(sigs, p_nodes, cor_th)
np.fill_diagonal(A, 0)
Asum = A.sum(axis=0)
none_iso_nodes, = Asum.nonzero()

shrink_map = dict(zip(range(len(none_iso_nodes)), none_iso_nodes))
A = A[np.ix_(none_iso_nodes, none_iso_nodes)]

inta = inta[none_iso_nodes]
node_num = A.shape[0]


print('--> start to generate csdp problem')
P0, q0, W = com_det_reg(A, inta, w1, w2, lamb, out='./prob.sdpb')

print('--> start to solve csdp problem')
# Please change the CSDP_bin to an apporiate value if check_call fails.
CSDP_bin = 'csdp'
check_call([CSDP_bin, './prob.sdpb', './botnet.sol'])

print('--> start to parse csdp solution')
Z, X = parse_CSDP_sol('botnet.sol', node_num + 1)
solution = randomization(X, P0, q0, sn=10000)

botnet_nodes_set = set(botnet_nodes)
ref_sol = []
for nv in none_iso_nodes:
    ip = mix_nodes[nv]
    if ip in botnet_nodes_set:
        ref_sol.append(1)
    else:
        ref_sol.append(0)

def eval_obj(fea_sol):
    fea_sol = np.asarray(fea_sol)
    return np.dot(np.dot(fea_sol.T, P0), fea_sol) + np.dot(q0, fea_sol)

import igraph
ig = igraph.Graph(directed=False)
ig.add_vertices(range(len(shrink_map)))
I, J = A.nonzero()
edges = zip(I, J)
ig.add_edges(edges)
res = ig.community_leading_eigenvector(clusters=2)
solution3 = res.membership


e1 = eval_obj(solution)
e3 = eval_obj(solution3)
e_ref = eval_obj(ref_sol)
print('SDP Solution: %f\nleading eigen value: %f\nreference: %f\n'
        % (e1, e3, e_ref))

inta_diff = np.dot(inta, solution)
print('inta_diff', inta_diff)


botnet, = np.nonzero(solution > 0)
print('[%i] ips out of [%i] ips are detected as bots' %
      (len(botnet), node_num))


tr = dict(A=A, solution=solution, mix_nodes=mix_nodes,
        botnet_nodes=botnet_nodes, shrink_map=shrink_map,
        ref_sol=ref_sol, inta=inta)

zdump(tr, 'viz-com-test.pkz')