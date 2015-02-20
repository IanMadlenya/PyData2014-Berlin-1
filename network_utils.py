import numpy as np

import networkx as nx
import planarity

from graph.algos import construct_pmfg

def construct_pmfg(df_corr_matrix):
    # TW: Is this required? What's the difference to graph.algos.construct_pmfg?
    df_distance = np.sqrt( 2 * np.clip( 1. - df_corr_matrix, 0., 2.) ) 
    dist_matrix = df_distance.values
    index_upper_triangular = np.triu_indices(dist_matrix.shape[0],1)
    isort = np.argsort( dist_matrix[index_upper_triangular] )
    G = nx.Graph()
    for k in xrange(0,len(isort)):
        u = index_upper_triangular[0][isort[k]]
        v = index_upper_triangular[1][isort[k]]
        if dist_matrix[u,v] > 0: # remove perfect correlation because of diagonal FIXME
            G.add_edge(u, v, {'weight': float(dist_matrix[u,v])})
            if not planarity.is_planar(G):
                G.remove_edge(u,v)
    return G

def construct_mst(df, threshold=0.01):
    # threshold = 0.05 / len(names):
    g = nx.Graph()
    names = df.columns.unique()
    df_distance = np.sqrt( 2 * np.clip( 1. - df, 0., 2.) ) 
    for i, n in enumerate(names):
        for j, m in enumerate(names):
            if j >= i: break
            val = df_distance.loc[n,m]
            # Bonferroni correction: TODO check implementation
            if np.abs(val) > threshold:
                g.add_edge(n, m, weight=val)
    return nx.minimum_spanning_tree(g)