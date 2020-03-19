import glob
import argparse
import collections
import string,json, os, sys
import networkx as nx
import random
import os
import numpy as np
import math

import operator

# import snap

def gen_ForestFire(n, filename_nodes, filename_edges, forward_p=0.5, backward_p=0.5, p_dem=0.3, p_rep=0.3, p_neut=0.3, p_across_dem_rep=0.3,p_across_dem_neut=0.3,
                              p_across_rep_dem=0.3, p_across_rep_neut=0.3, p_across_neut_dem=0.3, p_across_neut_rep=0.3,
                              w_d_d=[0.3,0.3], w_d_r=[0.3,0.3], w_d_n=[0.3,0.3],w_r_d=[0.3,0.3], w_r_r=[0.3,0.3],
                              w_r_n=[0.3,0.3],w_n_d=[0.3,0.3], w_n_r=[0.3,0.3], w_n_n=[0.3,0.3],
                              group_ratio=[0.33,0.66]):
    
    # G = snap.GenForestFire(n, forward_p, backward_p)
    G = snap.GenPrefAttach(1000, 10)
    # for node in G.nodes():
    print(G.GetEdges())
    # for node in G.Nodes():
    #     print(str(node.GetId()) + ': ' + str(node.GetOutDeg()))
        # for e in range(node.GetOutDeg()):
        #     node.GetOutNId(e)
    # for EI in G.Edges():
    #     print("edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))


def gen_graph_divisive():
    filename_nodes=""
    filename_edges=""
    p_dem = ave_deg*0.7649; p_rep = ave_deg*0.8486; p_neut = ave_deg*0.4341;
    p_across_dem_rep = ave_deg*0.0354; p_across_dem_neut = ave_deg*0.1998;
    p_across_rep_dem = ave_deg*0.046; p_across_rep_neut = ave_deg*0.1054
    p_across_neut_dem = ave_deg*0.293; p_across_neut_rep = ave_deg*0.273; group_ratio = [0.33, 0.66]


####w-->weight, d-->dem, r-->rep, n-->neut, h-->high-con,l-->low-cons
    w_d_h_d= .65 ; w_d_l_d= .77; w_d_h_r=0.077; w_d_l_r=0.04; w_d_h_n=.27;w_d_l_n=0.19
    w_d_d=[w_d_h_d, w_d_l_d]; w_d_r=[w_d_h_r,w_d_l_r]; w_d_n=[w_d_h_n, w_d_l_n]
    w_r_h_d= .1 ; w_r_l_d= .08; w_r_h_r=0.72; w_r_l_r=0.76; w_r_h_n=.18;w_r_l_n=0.14
    w_r_d=[w_r_h_d, w_r_l_d]; w_r_r=[w_r_h_r, w_r_l_r]; w_r_n=[w_r_h_n, w_r_l_n]
    w_n_h_d= .6 ; w_n_l_d= .68; w_n_h_r=0.05; w_n_l_r=0.08; w_n_h_n=.35;w_n_l_n=0.23
    w_n_d=[w_n_h_d, w_n_l_d]; w_n_r=[w_n_h_r, w_n_l_r]; w_n_n=[w_n_h_n,w_n_l_n]
    node_clr, G = generatGraph_our_divisive(n, filename_nodes, filename_edges, p_dem, p_rep, p_neut, p_across_dem_rep, p_across_dem_neut,
                              p_across_rep_dem, p_across_rep_neut, p_across_neut_dem, p_across_neut_rep,
                              w_d_d, w_d_r, w_d_n,
                              w_r_d, w_r_r, w_r_n,
                              w_n_d, w_n_r, w_n_n,
                              group_ratio)

    return node_clr, G

def diffusion_old(node_clr, G, dem_source_news, dem_h_l_news, rep_source_news, rep_h_l_news, neut_source_news, neut_h_l_news):
    degr = G.degree(list(node_clr.keys()))
    degr_s = sorted(degr, key=operator.itemgetter(1), reverse=True)

    dem_set=[]
    rep_set=[]
    neut_set=[]
    ####select the nodes with highest degree from each group
    # for tup in degr_s:
    #     if len(dem_set)<dem_source_news or len(rep_set) < rep_source_news or len(neut_set) < neut_source_news:
    #         if node_clr[tup[0]]=='blue':
    #             if len(dem_set) <= dem_source_news:
    #                 dem_set.append(tup[0])
    #
    #         if node_clr[tup[0]]=='red':
    #             if len(rep_set) <= rep_source_news:
    #                 rep_set.append(tup[0])
    #         if node_clr[tup[0]]=='purple':
    #             if len(neut_set) <= neut_source_news:
    #                 neut_set.append(tup[0])
    #     else:
    #         break





    for tup in degr_s:
        if len(dem_set)<dem_source_news or len(rep_set) < rep_source_news or len(neut_set) < neut_source_news:
            if node_clr[tup[0]]=='blue':
                if len(dem_set) <= dem_source_news:
                    dem_set.append(tup[0])

            if node_clr[tup[0]]=='red':
                if len(rep_set) <= rep_source_news:
                    rep_set.append(tup[0])
            if node_clr[tup[0]]=='purple':
                if len(neut_set) <= neut_source_news:
                    neut_set.append(tup[0])
        else:
            break


    ###spread high cons news
    inf_list=collections.defaultdict()
    inf_list['dem']=collections.defaultdict(list)
    inf_list['rep']=collections.defaultdict(list)
    inf_list['neut']=collections.defaultdict(list)
    set_name=['dem', 'rep', 'neut']
    set_num=0
    num_news = [dem_h_l_news[0],rep_h_l_news[0], neut_h_l_news[0]]
    for set_s in [dem_set, rep_set, neut_set]:
        # for itr in range(dem_h_l_news[0]):
        res = [random.randrange(1, len(set_s), 1) for i in range(num_news[set_num])]
        for ind in res:
            for node_i in G.neighbors(set_s[ind]):
                    p_spread = G[set_s[ind]][node_i]['weight'][0]
                    Y = np.random.binomial(1, p_spread, 1)[0]
                    if Y == 1:
                        inf_list[set_name[set_num]][set_s[ind]].append(node_i)

        set_num += 1





    ###spread low cons news
    inf_list_l=collections.defaultdict()
    inf_list_l['dem']=collections.defaultdict(list)
    inf_list_l['rep']=collections.defaultdict(list)
    inf_list_l['neut']=collections.defaultdict(list)
    set_name=['dem', 'rep', 'neut']
    set_num=0
    num_news = [dem_h_l_news[1],rep_h_l_news[1], neut_h_l_news[1]]
    for set_s in [dem_set, rep_set, neut_set]:
        # for itr in range(dem_h_l_news[0]):
        res = [random.randrange(1, len(set_s), 1) for i in range(num_news[set_num])]

        for ind in res:
            for node_i in G.neighbors(set_s[ind]):
                    p_spread = G[set_s[ind]][node_i]['weight'][1]
                    Y = np.random.binomial(1, p_spread, 1)[0]
                    if Y == 1:
                        inf_list_l[set_name[set_num]][set_s[ind]].append(node_i)

        set_num += 1

    return inf_list, inf_list_l


def diffusion(node_clr, G, dem_source_news, dem_h_l_news, rep_source_news, rep_h_l_news, neut_source_news, neut_h_l_news):
    degr = G.degree(list(node_clr.keys()))
    degr_s = sorted(degr, key=operator.itemgetter(1), reverse=True)

    dem_set_h=[]
    rep_set_h=[]
    neut_set_h=[]
    dem_set_l=[]
    rep_set_l=[]
    neut_set_l=[]
    ####select the nodes with highest degree from each group
    for tup in degr_s:
        if len(dem_set_h)<dem_h_l_news[0] or len(rep_set_h) < rep_h_l_news[0] or len(neut_set_h) < neut_h_l_news[0]:
            if node_clr[tup[0]]=='blue':
                if len(dem_set_h) <= dem_source_news:
                    dem_set_h.append(tup[0])

            if node_clr[tup[0]]=='red':
                if len(rep_set_h) <= rep_source_news:
                    rep_set_h.append(tup[0])
            if node_clr[tup[0]]=='purple':
                if len(neut_set_h) <= neut_source_news:
                    neut_set_h.append(tup[0])
        else:
            break


    ####select the nodes with highest degree from each group
    for tup in degr_s:
        if len(dem_set_l)<dem_h_l_news[1] or len(rep_set_l) < rep_h_l_news[1] or len(neut_set_l) < neut_h_l_news[1]:
            if node_clr[tup[0]]=='blue':
                if len(dem_set_l) <= dem_source_news:
                    dem_set_l.append(tup[0])

            if node_clr[tup[0]]=='red':
                if len(rep_set_l) <= rep_source_news:
                    rep_set_l.append(tup[0])
            if node_clr[tup[0]]=='purple':
                if len(neut_set_l) <= neut_source_news:
                    neut_set_l.append(tup[0])
        else:
            break



    inf_list=collections.defaultdict()
    inf_list['dem']=collections.defaultdict(list)
    inf_list['rep']=collections.defaultdict(list)
    inf_list['neut']=collections.defaultdict(list)
    set_name=['dem', 'rep', 'neut']
    set_num=0
    # num_news = [dem_h_l_news[0],rep_h_l_news[0], neut_h_l_news[0]]
    for set_s in [dem_set_h, rep_set_h, neut_set_h]:
        # for itr in range(dem_h_l_news[0]):
        res = set_s
        for ind in res:
            for node_i in G.neighbors(ind):
                    p_spread = G[ind][node_i]['weight'][0]
                    Y = np.random.binomial(1, p_spread, 1)[0]
                    if Y == 1:
                        inf_list[set_name[set_num]][ind].append(node_i)

        set_num += 1


    ###spread low cons news
    inf_list_l=collections.defaultdict()
    inf_list_l['dem']=collections.defaultdict(list)
    inf_list_l['rep']=collections.defaultdict(list)
    inf_list_l['neut']=collections.defaultdict(list)
    set_name=['dem', 'rep', 'neut']
    set_num=0
    num_news = [dem_h_l_news[1],rep_h_l_news[1], neut_h_l_news[1]]
    for set_s in [dem_set_l, rep_set_l, neut_set_l]:
        # for itr in range(dem_h_l_news[0]):
        res = set_s

        for ind in res:
            for node_i in G.neighbors(ind):
                    p_spread = G[ind][node_i]['weight'][1]
                    Y = np.random.binomial(1, p_spread, 1)[0]
                    if Y == 1:
                        inf_list_l[set_name[set_num]][ind].append(node_i)

        set_num += 1

    return inf_list, inf_list_l

def read_graph(filename_nodes, filename_edges):
    G = nx.Graph()
    node_clr={}
    f_nodes = open(filename_nodes,'r')
    f_edges = open(filename_edges,'r')
    for line in f_nodes:
        line=line.replace('\n','')
        l_s=line.split(' ')
        v = int(l_s[0])
        clr = l_s[1]
        node_clr[v] = clr
        G.add_node(v, color= clr, active=0, t=0)

    for line in f_edges:
        line=line.replace('\n','')
        l_s=line.split(' ')
        v1 = int(l_s[0])
        v2 = int(l_s[1])
        w1=l_s[2].replace('[','')
        w1=w1.replace(',','')
        w2=l_s[3].replace(']','')
        G.add_edges_from([(v1, v2)])
        G[v1][v2]['weight'] =[float(w1),float(w2)]

    return node_clr, G

def generatGraph_our_divisive(n, filename_nodes, filename_edges, p_dem=0.3, p_rep=0.3, p_neut=0.3, p_across_dem_rep=0.3,p_across_dem_neut=0.3,
                              p_across_rep_dem=0.3, p_across_rep_neut=0.3, p_across_neut_dem=0.3, p_across_neut_rep=0.3,
                              w_d_d=[0.3,0.3], w_d_r=[0.3,0.3], w_d_n=[0.3,0.3],
                              w_r_d=[0.3,0.3], w_r_r=[0.3,0.3],w_r_n=[0.3,0.3],
                              w_n_d=[0.3,0.3], w_n_r=[0.3,0.3], w_n_n=[0.3,0.3],
                              group_ratio=[0.33,0.66]):
    G = nx.Graph()
    node_clr={}
    #creating 3 clusters of nodes dem, rep, neut
    red_c=0
    blue_c=0
    neut_c=0
    for i in np.arange(n):

        toss = np.random.uniform(0, 1.0, 1)[0]
        # print(i)
        if toss <= group_ratio[0]:
            # Group rep
            G.add_node(i, color='red', active=0, t=0)
            node_clr[i] = 'red'
            red_c += 1
        elif toss <= group_ratio[1]:
            # Group dem
            G.add_node(i, color='blue', active=0, t=0)
            node_clr[i]='blue'
            blue_c+=1

        else:
            # Group neut
            G.add_node(i, color='purple', active=0, t=0)
            node_clr[i]='purple'
            neut_c+=1
    num_edges = 0
    for i in np.arange(n):
        for j in np.arange(n):
            if G.has_edge(i, j) or i == j:
                continue

            if G.nodes[i]['color'] == G.nodes[j]['color']:
                if G.nodes[i]['color'] == 'blue':
                    p_with = p_dem
                    link_weight = w_d_d
                elif G.nodes[i]['color'] == 'red':
                    p_with = p_rep
                    link_weight = w_r_r
                else:
                    p_with = p_neut
                    link_weight = w_n_n

                Y = np.random.uniform(0, 1.0, 1)[0]
                if Y<p_with:
                    Y=1
                else:
                    Y=0
                # Y = np.random.binomial(1, p_with, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i, j)])
                    G[i][j]['weight'] = link_weight
                    # G[i][j]['weight'] = np.random.uniform(0,0.1,1)[0]
                    num_edges += 1

            elif G.nodes[i]['color'] == 'blue':
                if G.nodes[j]['color']=='red':
                    p_across = p_across_dem_rep
                    link_weight = w_d_r
                elif G.nodes[j]['color'] == 'purple':
                    p_across = p_across_dem_neut
                    link_weight = w_d_n

                Y = np.random.uniform(0, 1.0, 1)[0]
                if Y<p_across:
                    Y=1
                else:
                    Y=0
                # Y = np.random.binomial(1, p_across, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i, j)])
                    G[i][j]['weight'] =  link_weight
                    num_edges += 1

            elif G.nodes[i]['color'] == 'red':
                if G.nodes[j]['color']=='blue':
                    p_across = p_across_rep_dem
                    link_weight = w_r_d
                elif G.nodes[j]['color'] == 'purple':
                    p_across = p_across_rep_neut
                    link_weight = w_r_n
                Y = np.random.uniform(0, 1.0, 1)[0]
                if Y<p_across:
                    Y=1
                else:
                    Y=0
                # Y = np.random.binomial(1, p_across, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i, j)])
                    G[i][j]['weight'] = link_weight
                    num_edges += 1

            else:
                if G.nodes[j]['color']=='blue':
                    p_across = p_across_neut_dem
                    link_weight = w_n_d
                elif G.nodes[j]['color'] == 'red':
                    p_across = p_across_neut_rep
                    link_weight = w_n_r
                Y = np.random.uniform(0, 1.0, 1)[0]
                if Y<p_across:
                    Y=1
                else:
                    Y=0

                # Y = np.random.binomial(1, p_across, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i, j)])
                    G[i][j]['weight'] = link_weight
                    num_edges += 1

    nx.write_weighted_edgelist(G, filename_edges)
    print('number of edges: ' + str(len(G.edges())))

    if filename_nodes:
        with open(filename_nodes, 'w+') as f:
            for v in G.nodes():
                f.write('%s %s%s' % (v, G.nodes._nodes[v]['color'], os.linesep))



    return node_clr, G

def generateGraph(n, m, filename='', pw=.75, maxw=5):
    G = nx.dense_gnm_random_graph(n, m)
    for e in G.edges():
        if random.random() < pw:
            G[e[0]][e[1]]['weight'] = [1]
        else:
            G[e[0]][e[1]]['weight'] = random.randint(2, maxw)
    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' % (len(G.nodes()), len(G.edges()), os.linesep))
            for v1, v2, edata in G.edges(data=True):
                # for it in range(edata['weight']):
                f.write('%s %s %s%s' % (v1, v2, edata['weight'], os.linesep))
    return G

def my_add_function(a, b):
    c = a+b
    return c

def generateGraph_ours(n, m, filename='', p_cliq=.75):
    # DG = nx.DiGraph()
    G = nx.Graph()
    nodes_dem = []
    nodes_rep = []
    nodes_neut = []
    for i in np.arange(n):

        toss = np.random.random_sample()
        if toss >= float(2)/3:
            G.add_nodes_from(i, color='blue', active=0, t=0)
            nodes_dem.append(i)
        elif toss>=float(1)/3:
            G.add_nodes_from(i, color='red', active=0, t=0)
            nodes_rep.append(i)
        else:
            G.add_nodes_from(i, color='purple', active=0, t=0)
            nodes_neut.append(i)

    i = 0
    while i < m:
        Y = np.random.binomial(1, p_cliq, 1)
        n_1 = np.random.randint(0, n)

        if n_1 in nodes_a:
            if Y == 1:
                n_2 = nodes_a[np.random.randint(0, len(nodes_a))]

            else:
                n_2 = nodes_b[np.random.randint(0, len(nodes_b))]
        else:
            if Y == 1:
                n_2 = nodes_b[np.random.randint(0, len(nodes_b))]
            else:
                n_2 = nodes_a[np.random.randint(0, len(nodes_a))]

        if G.has_edge(n_1, n_2) or n_1 == n_2:
            continue

        G.add_edges_from([(n_1, n_2)])
        a = 0.0
        b = 0.5
        G[n_1][n_2]['weight'] = (b - a) * np.random.random_sample() + a
        i += 1

    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' % (len(G.nodes()), len(G.edges()), os.linesep))
            for v1, v2, edata in G.edges(data=True):
                for it in range(edata['weight']):
                    f.write('%s %s%s' % (v1, v2, os.linesep))
    return G


def generateGraphNPP(n, filename='', p_with=.75, p_across=0.1, group_ratio=0.7):
    # DG = nx.DiGraph()
    G = nx.Graph()

    for i in np.arange(n):

        toss = np.random.uniform(0, 1.0, 1)[0]
        # print(i)
        if toss <= group_ratio:
            G.add_node(i, color='red', active=0, t=0)

        else:
            G.add_node(i, color='blue', active=0, t=0)
    num_edges = 0
    for i in np.arange(n):
        for j in np.arange(n):
            if G.has_edge(i, j) or i == j:
                continue

            if G.nodes[i]['color'] == G.nodes[j]['color']:
                Y = np.random.binomial(1, p_with, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i, j)])
                    G[i][j]['weight'] = np.random.uniform(0, 0.1, 1)[0]
                    num_edges += 1

            else:
                Y = np.random.binomial(1, p_across, 1)[0]
                if Y == 1:
                    G.add_edges_from([(i, j)])
                    G[i][j]['weight'] = np.random.uniform(0, .1, 1)[0]
                    num_edges += 1

    print('number of edges: {num_edges}')

    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' % (len(G.nodes()), len(G.edges()), os.linesep))
            for n, ndata in G.nodes(data=True):
                f.write('%s %s%s' % (n, ndata['color'], os.linesep))
            for v1, v2, edata in G.edges(data=True):
                # for it in range(edata['weight']):
                f.write('%s %s %s%s' % (v1, v2, edata['weight'], os.linesep))

    return G


if __name__ == '__main__':
    # result = my_add_function(4, 6)
    # print(result)
    # generateGraph(30, 120, 'small_graph.txt')

    if len(sys.argv) == 1:
        sys.argv += ["-t", "generate_graph"]
        # sys.argv += ["-t", "generate_graph_ff"]
        # sys.argv += ["-t", "spread_high_low_cons_news"]
        # sys.argv += ["-t", "analysing_spread_news"]


    parser = argparse.ArgumentParser()
    parser.add_argument('inputpaths', nargs='*',
                        default=glob.glob("tweets-bilal/20*.gz"),
                        help='paths to gzipped files with tweets in json format')
    parser.add_argument('-t',
                        default="extract-userinfo-usercrawl",
                        help='task name')
    parser.add_argument('-c',
                        default="set3-en",
                        # default="set2-all",
                        # default="nov3m",
                        # default="all",
                        help='crawlname')
    parser.add_argument('-ct',
                        # default="tweets-retweeters",
                        default="tweets-expall",
                        help='crawltype')
    parser.add_argument('-usn',
                        default="set3-en",
                        help="usersetnm")
    parser.add_argument('-muzzled',
                        default=False,
                        action='store_true')
    parser.add_argument('-nj',
                        default=None,
                        help='n_jobs_remote')
    parser.add_argument('-pn',
                        default="before-feb2016-1m",
                        help='periodname')
    parser.add_argument('-f',
                        default="0",
                        help='num_followers_bin')

    args = parser.parse_args()
    crawlname = os.path.basename(args.c)
    crawltype = os.path.basename(args.ct)
    usersetnm = os.path.basename(args.usn)
    periodname = os.path.basename(args.pn)
    f_g_bin = int(args.f)
    # print(f_g_bin)
    inputpaths = args.inputpaths



    # p_zero = 0.03
    # p_one = 0.03
    # p_across = 0.000001
    # group_ratio = 0.49
    # GRAPH_SIZE = 1200
    # n= GRAPH_SIZE
    # total: 2214467
    # n = 100
    n = 1000
    # n = 10000
    # n = 100000
    ave_deg = 0.004
    # ave_deg = 1
    # filename_nodes = 'graphs/test_n'
    # filename_edges = 'graphs/test_e'
    filename_nodes = 'graphs/nodes_'+str(n)
    filename_edges = 'graphs/edges_'+str(n)
    # filename_nodes = 'graphs/ffnodes_'+str(n)
    # filename_edges = 'graphs/ffedges_'+str(n)
    dem_source_news = 4; dem_h_l_news = [45, 63]
    rep_source_news = 4; rep_h_l_news = [32, 26]
    neut_source_news = 2; neut_h_l_news = [23, 11]

    #
    # dem_source_news = 4; dem_h_l_news = [45, 45]
    # rep_source_news = 4; rep_h_l_news = [45, 45]
    # neut_source_news = 2; neut_h_l_news = [45, 45]

    if args.t == "spread_high_low_cons_news":



        node_clr, G = read_graph(filename_nodes, filename_edges)


        audience_count = collections.defaultdict(list)
        audience_count_h_l= collections.defaultdict(list)
        ret_foll_dict_global=collections.defaultdict(list)
        out_dict_global_h=collections.defaultdict(list)
        out_dict_global_l=collections.defaultdict(list)
        for itr in range(10):
            print('iteration: ' + str(itr))
            # node_clr, G=gen_graph_divisive()
            inf_list, inf_list_l= diffusion_old(node_clr, G, dem_source_news, dem_h_l_news, rep_source_news, rep_h_l_news, neut_source_news, neut_h_l_news)

            ret_foll = collections.defaultdict(list)
            ret_foll_h_l= collections.defaultdict(list)
            ret_foll_dict_h=collections.defaultdict(dict)
            ret_foll_dict_l=collections.defaultdict(dict)
            for source_leaning in inf_list:
                # if source_leaning not in ret_foll_dict:
                ret_foll_dict_h[source_leaning]=collections.defaultdict(list)
                for t_id in inf_list[source_leaning]:
                    for foll_id in inf_list[source_leaning][t_id]:
                        ret_foll_h_l['source_'+str(source_leaning) + '_h_' + str(node_clr[foll_id])].append(foll_id)
                        ret_foll['source_'+str(source_leaning)  + '_'+ str(node_clr[foll_id])].append(foll_id)
                        ret_foll_dict_h[source_leaning][node_clr[foll_id]].append(foll_id)

            for key in ret_foll:
                audience_count[key].append(len(ret_foll[key]))

            # for key1 in ret_foll_dict:
            #     ret_foll_dict_global[key1]=collections.defaultdict(list)
            #     for key2 in ret_foll_dict[key1]:
            #         ret_foll_dict_global[key1][key2].append(len(ret_foll_dict[key1][key2]))
            # print('Low Consensus news##############')
            ret_foll = collections.defaultdict(list)

            for source_leaning in inf_list_l:
                ret_foll_dict_l[source_leaning]=collections.defaultdict(list)
                for ret_id in inf_list_l[source_leaning]:
                    for foll_id in inf_list_l[source_leaning][ret_id]:
                        ret_foll_h_l['source_'+str(source_leaning) + '_l_' + str(node_clr[foll_id])].append(foll_id)
                        ret_foll['source_'+str(source_leaning)  + '_'+ str(node_clr[foll_id])].append(foll_id)
                        ret_foll_dict_l[source_leaning][node_clr[foll_id]].append(foll_id)

            for key in ret_foll:
                audience_count[key].append(len(ret_foll[key]))
            for key in ret_foll_h_l:
                    audience_count_h_l[key].append(len(ret_foll_h_l[key]))



            for source_lean in ret_foll_dict_h:
                out_dict = collections.defaultdict()
                sum=0
                for ret_lean in ret_foll_dict_h[source_lean]:
                    out_dict[str(source_lean)+'_'+str(ret_lean)]=len(ret_foll_dict_h[source_lean][ret_lean])
                    sum+=len(ret_foll_dict_h[source_lean][ret_lean])
                for key in out_dict:
                    out_dict[key] =out_dict[key]/sum
                    out_dict_global_h[key].append(out_dict[key])


            for source_lean in ret_foll_dict_l:
                out_dict = collections.defaultdict()
                sum=0
                for ret_lean in ret_foll_dict_l[source_lean]:
                    out_dict[str(source_lean)+'_'+str(ret_lean)]=len(ret_foll_dict_l[source_lean][ret_lean])
                    sum+=len(ret_foll_dict_l[source_lean][ret_lean])
                for key in out_dict:
                    out_dict[key] =out_dict[key]/sum
                    out_dict_global_l[key].append(out_dict[key])


        print('retweeters#############################')
        for key in audience_count_h_l:
            print(str(key) + ' number: ' + str(np.mean(audience_count_h_l[key])))

        print('retweeters high#############################')
        for key in out_dict_global_h:
            print(str(key) + ' number: ' + str(np.mean(out_dict_global_h[key])))

        print('retweeters low#############################')
        for key in out_dict_global_l:
            print(str(key) + ' number: ' + str(np.mean(out_dict_global_l[key])))


        print('retweeters#############################')
        # for key in audience_count_h_l:
        #     print(str(key) + ' number: ' + str(np.mean(audience_count_h_l[key])))


        print('followers of retweeters#############################')



    if args.t == "generate_graph_ff":
        ave_deg = .33

        p_dem = ave_deg*0.7649; p_rep = ave_deg*0.8486; p_neut = ave_deg*0.4341;
        p_across_dem_rep = ave_deg*0.0354; p_across_dem_neut = ave_deg*0.1998;
        p_across_rep_dem = ave_deg*0.046; p_across_rep_neut = ave_deg*0.1054
        p_across_neut_dem = ave_deg*0.293; p_across_neut_rep = ave_deg*0.273; group_ratio = [0.33, 0.66]


    ####w-->weight, d-->dem, r-->rep, n-->neut, h-->high-con,l-->low-cons
        w_d_h_d= .65 ; w_d_l_d= .77; w_d_h_r=0.077; w_d_l_r=0.04; w_d_h_n=.27;w_d_l_n=0.19
        w_d_d=[w_d_h_d, w_d_l_d]; w_d_r=[w_d_h_r,w_d_l_r]; w_d_n=[w_d_h_n, w_d_l_n]
        w_r_h_d= .1 ; w_r_l_d= .08; w_r_h_r=0.72; w_r_l_r=0.76; w_r_h_n=.18;w_r_l_n=0.14
        w_r_d=[w_r_h_d, w_r_l_d]; w_r_r=[w_r_h_r, w_r_l_r]; w_r_n=[w_r_h_n, w_r_l_n]
        w_n_h_d= .6 ; w_n_l_d= .68; w_n_h_r=0.05; w_n_l_r=0.08; w_n_h_n=.35;w_n_l_n=0.23
        w_n_d=[w_n_h_d, w_n_l_d]; w_n_r=[w_n_h_r, w_n_l_r]; w_n_n=[w_n_h_n,w_n_l_n]
        # generatGraph_our_divisive(n, filename_nodes, filename_edges)
        forward_p=ave_deg; backward_p=ave_deg
        gen_ForestFire(n, filename_nodes, filename_edges, forward_p, backward_p, p_dem, p_rep, p_neut, p_across_dem_rep, p_across_dem_neut,
                                  p_across_rep_dem, p_across_rep_neut, p_across_neut_dem, p_across_neut_rep,
                                  w_d_d, w_d_r, w_d_n,
                                  w_r_d, w_r_r, w_r_n,
                                  w_n_d, w_n_r, w_n_n,
                                  group_ratio)


    if args.t == "generate_graph":

        # p_dem = ave_deg*0.7249; p_rep = ave_deg*0.74086; p_neut = ave_deg*0.4341;
        # p_across_dem_rep = ave_deg*0.0554; p_across_dem_neut = ave_deg*0.1998;
        # p_across_rep_dem = ave_deg*0.076; p_across_rep_neut = ave_deg*0.1654
        # p_across_neut_dem = ave_deg*0.293; p_across_neut_rep = ave_deg*0.273; group_ratio = [0.34, 0.67]

        p_dem = ave_deg*0.7649; p_rep = ave_deg*0.8486; p_neut = ave_deg*0.4341;
        p_across_dem_rep = ave_deg*0.0354; p_across_dem_neut = ave_deg*0.1998;
        p_across_rep_dem = ave_deg*0.046; p_across_rep_neut = ave_deg*0.1054
        p_across_neut_dem = ave_deg*0.293; p_across_neut_rep = ave_deg*0.273; group_ratio = [0.33, 0.66]

    ####w-->weight, d-->dem, r-->rep, n-->neut, h-->high-con,l-->low-cons
        w_d_h_d= .65 / 2; w_d_l_d= .85/ 2; w_d_h_r=0.077 / 0.1; w_d_l_r=0.04/ 0.1; w_d_h_n=.27/7;w_d_l_n=0.11/7
        w_d_d=[w_d_h_d, w_d_l_d]; w_d_r=[w_d_h_r,w_d_l_r]; w_d_n=[w_d_h_n, w_d_l_n]
        w_r_h_d= .12 / 1.2; w_r_l_d= .08/1.2; w_r_h_r=0.68/5; w_r_l_r=0.85/5; w_r_h_n=.18/4;w_r_l_n=0.1/4
        w_r_d=[w_r_h_d, w_r_l_d]; w_r_r=[w_r_h_r, w_r_l_r]; w_r_n=[w_r_h_n, w_r_l_n]
        # w_n_h_d= .6 ; w_n_l_d= .68; w_n_h_r=0.05; w_n_l_r=0.08; w_n_h_n=.35;w_n_l_n=0.23
        w_n_h_d= .3 ; w_n_l_d= .38; w_n_h_r=0.35; w_n_l_r=0.38; w_n_h_n=.35;w_n_l_n=0.23
        w_n_d=[w_n_h_d, w_n_l_d]; w_n_r=[w_n_h_r, w_n_l_r]; w_n_n=[w_n_h_n,w_n_l_n]
        # generatGraph_our_divisive(n, filename_nodes, filename_edges)
        generatGraph_our_divisive(n, filename_nodes, filename_edges, p_dem, p_rep, p_neut, p_across_dem_rep, p_across_dem_neut,
                                  p_across_rep_dem, p_across_rep_neut, p_across_neut_dem, p_across_neut_rep,
                                  w_d_d  , w_d_r, w_d_n,
                                  w_r_d, w_r_r, w_r_n,
                                  w_n_d, w_n_r, w_n_n,
                                  group_ratio)

        # generatGraph_our_divisive(n, filename_nodes, filename_edges, p_dem, p_dem, p_dem, p_across_dem_rep, p_across_dem_neut,
        #                           p_across_dem_rep, p_across_dem_neut, p_across_neut_dem, p_across_neut_rep,
        #                           # w_d_d, w_d_r, w_d_n,
        #                           # w_r_d, w_r_r, w_r_n,
        #                           w_d_d,w_d_r,w_d_n,
        #                           w_d_r,w_d_d,w_d_n,
        #                           w_n_d, w_n_r, w_n_n,
        #                           group_ratio)

        # ave deg = 0.033586889


    # if args.t == "analysing_spread_news":


    # ######################################################################################################
    # ######################################################################################################

    # source_dem_high_cons_dem: number 3428/45=76, 76/(76+9+32=117)= .65, average 0.10315197294150705
    # source_dem_high_cons_rep: number 395/45=9, 9/(76+9+32=117)= .077 , average - 0.14611488147610177
    # source_dem_high_cons_neut: number 1448/45=32, 32/(76+9+32=117)= .27, average 0.011170111809341724
    # source_dem_low_cons_dem: number 5933/63=94, 94/(94+5+23=122)= .77 , average 0.1224933582323342
    # source_dem_low_cons_rep: number 311/63=5, 5/(94+5+23=122)= .04, average - 0.1290679403081627
    # source_dem_low_cons_neut: number 1479/63=23,23/(94+5+23=122)= .19, average 0.011313887620351714
    # source_rep_high_cons_dem: number 285/32=9, 9/(9+65+14=88)= .1,  average 0.07107152336244701
    # source_rep_high_cons_rep: number 2090/32=65, 65/(9+65+14=88)= .72, average - 0.23379051827175615
    # source_rep_high_cons_neut: number 465/32=14, 14/(9+65+14=88)= .18,average 0.0027731072366406788
    # source_rep_low_cons_dem: number 321/26=12, 12/(12+114+24=150) = 0.08,average 0.07201247594342118
    # source_rep_low_cons_rep: number 2960/26=114, 114/(12+114+24=150) = .76,average - 0.2388612053084952
    # source_rep_low_cons_neut: number 627/26=24, 24/(12+114+24=150) = .14,average 0.001338854843453847
    # source_neut_high_cons_dem: number 2367/23=103, 103/(103+7+33=143)= .6,  average 0.09627698732889958
    # source_neut_high_cons_rep: number 169/23=7/(103+7+33=143)= 0.05, average - 0.16226899275508047
    # source_neut_high_cons_neut: number 757/23=33/(103+7+33=143)= 0.35, average 0.014088959093521954
    # source_neut_low_cons_dem: number 995/11=90, 90/(90+11+30=131) = .68, average 0.10373578646343005
    # source_neut_low_cons_rep: number 120/11=11, 11/(90+11+30) = 0.08,average - 0.177224018819505
    # source_neut_low_cons_neut: number 335/11=30, 30/(90+11+30) =  0.23,average 0.011812995266053937
    # ######################################################################################################
    # source_dem_high_cons_dem: total umber 45, number 45, average 0.10354566522506813
    # source_dem_high_cons_rep: total umber 45, number 45, average
    # source_dem_high_cons_neut: total umber 45, number 45, average
    # source_dem_low_cons_dem: total umber 63, number 63, average 0.13012692380273172
    # source_dem_low_cons_rep: total umber 63, number 63, average
    # source_dem_low_cons_neut: total umber 63, number 63, average
    # source_rep_high_cons_dem: total umber 32, number 32, average
    # source_rep_high_cons_rep: total umber 32, number 32, average - 0.22686010816204166
    # source_rep_high_cons_neut: total umber 32, number 32, average 0.003301229130181491
    # source_rep_low_cons_dem: total umber 26, number 26, average
    # source_rep_low_cons_rep: total umber 26, number 26, average - 0.23851186911883332
    # source_rep_low_cons_neut: total umber 26, number 26, average 0.00032989746118518976
    # source_neut_high_cons_dem: total umber 23, number 23, average 0.08927599457134314
    # source_neut_high_cons_rep: total umber 23, number 23, average
    # source_neut_high_cons_neut: total umber 23, number 23, average 0.01324312032227983
    # source_neut_low_cons_dem: total umber 11, number 11, average 0.09678955322795275
    # source_neut_low_cons_rep: total umber 11, number 11, average
    # source_neut_low_cons_neut: total umber 11, number 11, average 0.011723228127062736
    # ######################################################################################################
    # retweeter - dem_dem: number 2125977/(2125977+98264+555327=2779568) = 0.7649, ,2125977/182686=11.6373, ,  average 0.1595876036751991 --- >  2125977/5226919=0.4067
    # retweeter - dem_rep: number 98264/(2125977+98264+555327=2779568) = 0.0354, 98264/182686=0.5379, average - 0.14553452194822514--- >  98264/5226919=0.0188
    # retweeter - dem_neut: number 555327/(2125977+98264+555327=2779568) =0.1998 , /182686=3.0398, average 0.010591124627560653--- >  555327/5226919=0.1062
    # retweeter - rep_dem: number 88741/(88741+1635891+203232=1927864) = 0.046, 88741/59410=1.493704764, average 0.08766400015367566--- >  88741/5226919=0.017
    # retweeter - rep_rep: number 1635891/(88741+1635891+203232=1927864) = 0.8486, 1635891/59410=27.5356169, average - 0.269864602808309--- >  1635891/5226919=0.313
    # retweeter - rep_neut: number 203232/(88741+1635891+203232=1927864) = 0.1054, 203232/59410=3.4208, average 0.0037096346536492544--- >  203232/5226919=0.0389
    # retweeter - neut_dem: number 152195/(152195+141801+225491=519487)=0.293, 152195/185028=0.822551181, average 0.10760828794641726--- >  152195/5226919=0.0291
    # retweeter - neut_rep: number 141801/(152195+141801+225491=519487)=0.273, 141801/185028=0.7663759, average - 0.2310839244968664--- >  141801/5226919=0.0271
    # retweeter - neut_neut: number 225491/(152195+141801+225491=519487)=0.4341, 225491/185028=1.218685821, average 0.00876047370071181--- >  225491/5226919=0.0431
    # ######################################################################################################
    # retweeter - dem_dem: number 182686, average
    # retweeter - dem_rep: number 182686, average
    # retweeter - dem_neut: number 182686, average
    # retweeter - rep_dem: number 59410, average
    # retweeter - rep_rep: number 59410, average
    # retweeter - rep_neut: number 59410, average
    # retweeter - neut_dem: number 185028, average
    # retweeter - neut_rep: number 185028, average
    # retweeter - neut_neut: number 185028, average





    # dem - dem: 0.5995795427285976
    # dem - rep: 0.09214456237288784
    # dem - neut: 0.2782489607155365
    # rep - dem: 0.04368121385831113
    # rep - rep: 0.6786249505325487
    # rep - neut: 0.12190851791888403
    # neut - dem: 0.5995795427285976
    # neut - rep: 0.09214456237288784
    # neut - neut: 0.2782489607155365
    ###########################################simulatation#####################################
    ###########################################simulatation#####################################
    ###########################################simulatation#####################################
    ###########################################simulatation#####################################
    ###########################################simulatation#####################################
    ###########################################simulatation#####################################

    # retweeters  #############################
    # source_dem_blue number: 821.125 --> 821/(821+247+10=1078)=0.76
    # source_dem_purple number: 247.455-->247/(821+247+10=1078)=0.22
    # source_dem_red number: 10.12 --> 10/(821+247+10=1078)=0.12
    # source_rep_red number: 579.035 -->579/(579+16+4=598)=.96
    # source_rep_purple number: 16.695 -->16/(579+16+4=598)=.03
    # source_rep_blue number: 3.7766497461928936-->4/(579+16+4=598)=0.01
    # source_neut_purple number: 65.925 --> 66/(66+69+16=151)=0.43
    # source_neut_blue number: 68.705-->69/(66+69+16=151)=0.45
    # source_neut_red number: 16.495-->16/(66+69+16=151)=0.12
    # retweeters  #############################
    # source_dem_h_blue number: 683.22
    # source_dem_h_purple number: 207.61
    # source_dem_h_red number: 8.51
    # source_rep_h_red number: 638.27
    # source_rep_h_purple number: 18.63
    # source_rep_h_blue number: 4.09
    # source_neut_h_purple number: 89.25
    # source_neut_h_blue number: 92.12
    # source_neut_h_red number: 22.23
    # source_dem_l_blue number: 959.03
    # source_dem_l_purple number: 287.3
    # source_dem_l_red number: 11.73
    # source_rep_l_red number: 519.8
    # source_rep_l_purple number: 14.76
    # source_rep_l_blue number: 3.4536082474226806
    # source_neut_l_purple number: 42.6
    # source_neut_l_blue number: 45.29
    # source_neut_l_red number: 10.76

    # publisher  @Salon 16955991: fraction dem: 0.25, fraction rep: 0.01, fraction neut: 0.75
    # publisher @AP 51241574: fraction dem: 0.3333333333333333, fraction rep: 0.01, fraction neut: 0.6666666666666666
    # publisher @politico 9300262: fraction dem: 0.2857142857142857, fraction rep: 0.06593406593406594, fraction neut: 0.6483516483516484
    # publisher @CNN 759251: fraction dem: 0.13043478260869565, fraction rep: 0.01, fraction neut: 0.8695652173913043
    # avg: dem:0.25 , rep:0.01, neut:0.75
    # publisher @Slate 15164565: fraction dem: 0.0, fraction rep: 0.0, fraction neut: 1.0

    # publisher @nytimes 807095: fraction dem: 0.13186813186813187, fraction rep: 0.01098901098901099, fraction neut: 0.8571428571428571
    # publisher @Reuters 1652541: fraction dem: 0.0, fraction rep: 0.0, fraction neut: 1.0
    # avg: dem: 0.0.06, rep:0.0.005, neut:0.94

    # publisher @NEWS_MAKER 14669951: fraction dem: 0.10843373493975904, fraction rep: 0.5180722891566265, fraction neut: 0.37349397590361444
    # publisher  @BreitbartNews 457984599: fraction dem: 0.011363636363636364, fraction rep: 0.625, fraction neut: 0.36363636363636365
    # publisher @FoxNews 1367531: fraction dem: 0.15217391304347827, fraction rep: 0.043478260869565216, fraction neut: 0.8043478260869565
   #avg: dem: 0.12, rep:0.4, neut:0.5