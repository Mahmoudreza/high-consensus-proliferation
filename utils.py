import numpy as np
import networkx as nx

import sys
sys.path[:0] = ["/usr/local/lib/python2.7"]
# import matplotlib as mpl

# mpl.use('Agg')

# import matplotlib.pyplot as plt

# plt.ioff()
from operator import add
# from generateGraph import generateGraphNPP
import os
from networkx.algorithms import community
# from load_facebook_graph import *


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


# def load_graph(filename, p_with, p_across, group_ratio, num_nodes):
#     try:
#         f = open(filename + '.txt', 'r')
#         print("loaded: " + filename)
#         G = nx.Graph()
#         n, m = map(int, f.readline().split())
#         for i, line in enumerate(f):
#             if i < n:
#                 node_str = line.split()
#                 u = int(node_str[0])
#                 color = node_str[1]
#                 G.add_node(u, color=color)
#             else:
#                 edges_str = line.split()
#                 u = int(edges_str[0])
#                 v = int(edges_str[1])
#                 weight = float(edges_str[2])
#                 G.add_edge(u, v, weight=weight)
#         f.close()
#         # Store configuration file values
#     except FileNotFoundError:
#         print(f"File not found at {filename}, building new graph...")
#         G = generateGraphNPP(num_nodes, filename=filename + '.txt', p_with=p_with, p_across=p_across,
#                              group_ratio=group_ratio)
#
#     return G


def graph_stats(G, h_l=0, print_stats=True):
    # print average weights of each group
    w_r_within, w_b_within, w_n_within,w_across_r_b,w_across_r_n, w_across_b_r,w_across_b_n,w_across_n_r,w_across_n_b,\
    num_r, num_b, num_n, edges_r, edges_b, edges_n, edges_r_b,edges_r_n, edges_b_r, edges_b_n,\
    edges_n_r, edges_n_b= (0.0,) * 21
    for n, nbrs in G.adj.items():
        color = G.nodes[n]['color']
        if color == 'red':
            num_r += 1
        elif color == 'blue':
            num_b += 1
        else:
            num_n += 1

        for nbr, eattr in nbrs.items():
            if G.nodes[nbr]['color'] == color:
                if color == 'red':
                    w_r_within += eattr['weight'][h_l]
                    edges_r += 1
                elif color=='blue':
                    w_b_within += eattr['weight'][h_l]
                    edges_b += 1
                else:
                    w_n_within += eattr['weight'][h_l]
                    edges_n += 1

            elif G.nodes[nbr]['color'] == 'red':
                if color == 'blue':
                    w_across_r_b += eattr['weight'][h_l]
                    edges_r_b += 1
                elif color=='purple':
                    w_across_r_n += eattr['weight'][h_l]
                    edges_r_n += 1
            elif G.nodes[nbr]['color'] == 'blue':
                if color == 'red':
                    w_across_b_r += eattr['weight'][h_l]
                    edges_b_r += 1
                elif color == 'purple':
                    w_across_b_n += eattr['weight'][h_l]
                    edges_b_n += 1

            else :
                if color == 'red':
                    w_across_n_r += eattr['weight'][h_l]
                    edges_n_r += 1
                elif color == 'blue':
                    w_across_n_b += eattr['weight'][h_l]
                    edges_n_b += 1

    # for v1,v2,edata in G.edges(data=True):
    stats = {}
    stats['total_nodes'] = int(num_r + num_b + num_n)
    stats['group_r'] = int(num_r)
    stats['group_b'] = int(num_b)
    stats['group_n'] = int(num_n)
    total_edges = stats['total_edges'] = int(edges_r / 2 + edges_b / 2 + edges_r_b+ edges_r_n+ edges_b_r+ edges_b_n+ edges_n_b+ edges_n_r)
    stats['edges_group_r'] = int(edges_r / 2)
    stats['edges_group_b'] = int(edges_b / 2)
    stats['edges_group_n'] = int(edges_n / 2)
    stats['edges_r_n'] = int(edges_r_n)
    stats['edges_r_b'] = int(edges_r_b)
    stats['edges_b_n'] = int(edges_b_n)
    stats['edges_b_r'] = int(edges_b_r)
    stats['edges_n_r'] = int(edges_n_r)
    stats['edges_n_b'] = int(edges_n_b)


    stats['weights_group_r'] = w_r_within / edges_r
    stats['weights_group_b'] = w_b_within / edges_b
    stats['weights_group_n'] = w_n_within / edges_n
    # stats['weights_across'] = w_across / edges_across
    if print_stats:
        print(
            '\n \n Red Nodes: {num_r}, Blue Nodes: {num_b}, Purple Nodes: {num_n},'
            ' edges total = {total_edges} edges_within r: {edges_r / 2}, edges_within_b {edges_b / 2},edges_within_n {edges_n / 2},'
            ' edges_r_b {edges_r_b}, edges_r_n {edges_r_n}, edges_b_r {edges_b_r}, edges_b_n {edges_b_n}, edges_n_r {edges_n_r}, edges_n_b {edges_n_b}'
            ' average degree r: {edges_r / num_r}, average degree b: {edges_b / num_b},average degree n: {edges_n / num_n},'
            # f' weights within red {w_a_within / edges_a}, weights within b: {w_b_within / edges_b}, weights accross: {w_across / edges_across}'
            ' \n \n \n ')
    return stats


def write_files(filename, num_influenced, num_influenced_r, num_influenced_b, num_influenced_n, seeds_r, seeds_b, seeds_n):
    '''
    write num_influenced, num_influenced_a, num_influenced_b -> list
          and seeds_a list of lists i.e. actual id's of the seeds chosen
              seeds_b list of lists
    each row
    I I_r I_b I_n [seed_list_r comma separated];[seed_list_b];[seed_list_n]
    .
    .
    .
    '''
    f = open(filename + '_results.txt', 'w')
    for I, I_r, I_b, I_n, S_r, S_b, S_n in zip(num_influenced, num_influenced_r, num_influenced_b, num_influenced_n, seeds_r, seeds_b, seeds_n):
        f.write('{str(I)} {str(I_r)} {str(I_b)} {str(I_n)} ')
        for i, seed in enumerate(S_r):
            if i == len(S_r) - 1:
                f.write('{seed}')
            else:
                f.write('{seed},')
        f.write(';')
        for i, seed in enumerate(S_b):
            if i == len(S_b) - 1:
                f.write('{seed}')
            else:
                f.write('{seed},')
        f.write(';')
        for i, seed in enumerate(S_n):
            if i == len(S_n) - 1:
                f.write('{seed}')
            else:
                f.write('{seed},')
        f.write('\n')

    f.close()


def read_files(filename):
    '''
    returns num_influenced, num_influenced_a, num_influenced_b -> list
          and seeds_a list of lists i.e. actual id's of the seeds chosen
              seeds_b list of lists

    '''
    f = open(filename + '_results.txt', 'r')
    num_influenced = []
    num_influenced_r = []
    num_influenced_b = []
    num_influenced_n = []
    seeds_r = []
    seeds_b = []
    seeds_n = []

    for line in f:
        I, I_r, I_b, I_n, residue = line.split()
        num_influenced.append(float(I))
        num_influenced_r.append(float(I_r))
        num_influenced_b.append(float(I_b))
        num_influenced_n.append(float(I_n))

        S_r, S_b, S_n = residue.split(';')
        S_r_list = []

        if S_r != '':
            S_a_list = list(map(int, S_r.split(',')))
        seeds_r.append(S_a_list)

        S_b_list = []
        if S_b != '':
            S_b_list = list(map(int, S_b.split(',')))
        seeds_b.append(S_b_list)

        S_n_list = []
        if S_n != '':
            S_n_list = list(map(int, S_n.split(',')))
        seeds_n.append(S_n_list)

    f.close()
    return num_influenced, num_influenced_r, num_influenced_b, num_influenced_n, seeds_r, seeds_b, seeds_n


def plot_influence(influenced_a, influenced_b, influenced_c, num_seeds, filename, population_a, population_b,population_c, num_seeds_a,
                   num_seeds_b,num_seeds_c):
    # total influence
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, num_seeds + 1), list(map(add, influenced_a, list(map(add, influenced_b, influenced_c)))), 'g+')
    plt.xlabel('Number of Seeds')
    plt.ylabel('Total Influenced Nodes')
    # plt.legend(loc='best')
    plt.savefig(filename + '_total_influenced.png', bbox_inches='tight')
    plt.close(fig)
    # total influence fraction
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, num_seeds + 1),
             np.asarray(list(map(add, influenced_a, list(map(add, influenced_b, influenced_c))))) / (population_a + population_b+ population_c), 'g+')
    plt.xlabel('Number of Seeds')
    plt.ylabel('Total Fraction Influenced Nodes')
    # plt.legend(loc='best')
    plt.savefig(filename + '_total_fraction_influenced.png', bbox_inches='tight')
    plt.close(fig)
    # group wise influenced
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, num_seeds + 1), influenced_a, 'r+', label='Group Rep')
    plt.plot(np.arange(1, num_seeds + 1), influenced_b, 'b^', label='Group Dem')
    plt.plot(np.arange(1, num_seeds + 1), influenced_c, 'g*', label='Group Neut')
    plt.xlabel('Number of Seeds')
    plt.ylabel('Total Influenced Nodes')
    plt.legend(loc='best')
    plt.savefig(filename + '_group_influenced.png', bbox_inches='tight')
    plt.close(fig)

    # fraction group influenced
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, num_seeds + 1), np.asarray(influenced_a) / population_a, 'r+', label='Group Rep')
    plt.plot(np.arange(1, num_seeds + 1), np.asarray(influenced_b) / population_b, 'b^', label='Group Dem')
    plt.plot(np.arange(1, num_seeds + 1), np.asarray(influenced_c) / population_c, 'g*', label='Group Neut')
    plt.xlabel('Number of Seeds')
    plt.ylabel('Fraction of Influenced Nodes')
    plt.legend(loc='best')
    plt.savefig(filename + '_fraction_group_influenced.png', bbox_inches='tight')
    plt.close(fig)

    # comparison abs difference
    fig = plt.figure(figsize=(6, 4))
    index = np.arange(1, num_seeds + 1)
    plt.plot(index, np.abs(np.asarray(influenced_a) / population_a - np.asarray(influenced_b) / population_b)+
                np.abs(np.asarray(influenced_a) / population_a - np.asarray(influenced_c) / population_b)+
                np.abs(np.asarray(influenced_b) / population_a - np.asarray(influenced_c) / population_b),ls='-', alpha=0.5)

    plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.xlabel('Number of Seeds')
    plt.ylabel('Absolute difference of Influenced Nodes \n (|Fa - Fb| + |Fa - Fc| + |Fb - Fc| + )')
    plt.savefig(filename + '_difference_total_influenced.png', bbox_inches='tight')
    plt.close(fig)



    # Seeds group memeber ship
    # fig = plt.figure(figsize=(6,4))
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    index = np.arange(1, num_seeds + 1)
    rects1 = ax.bar(index, num_seeds_a, bar_width,
                    color='r',
                    label='Group Rep')
    print(num_seeds_a)
    rects2 = ax.bar(index + bar_width, num_seeds_b, bar_width,
                    color='b',
                    label='Group Dem')
    print(num_seeds_b)

    rects2 = ax.bar(index + bar_width+ bar_width, num_seeds_c, bar_width,
                    color='g',
                    label='Group Neut')
    plt.legend(loc='best')
    ax.set_xlabel('Total Number of Seeds')
    ax.set_ylabel('Number of Seeds from each group')
    ax.set_title('Seed distribution in groups')
    ax.set_xticks(index)  # + bar_width / 2)

    # plt.plot(np.arange(1, num_seeds + 1), num_seeds_a / num_seeds, 'r+')
    # plt.plot(np.arange(1, num_seeds + 1), num_seeds_b / num_seeds, 'b.')
    # plt.xlabel('Total Number of Seeds')
    # plt.ylabel('Number from each group')
    plt.savefig(filename + '_seed_groups.png', bbox_inches='tight')
    plt.close(fig)  #


def plot_influence_diff(influenced_a_list, influenced_b_list, influenced_c_list, num_seeds, labels, filename, population_a, population_b, population_c):
    '''
    list of lists influenced_a and influenced_b
    '''
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    index = np.arange(1, num_seeds + 1)
    for i, (influenced_a, influenced_b, influenced_c) in enumerate(zip(influenced_a_list, influenced_b_list, influenced_c_list)):
        ax.plot(index, (np.asarray(influenced_a) + np.asarray(influenced_b)+ np.asarray(influenced_c)) / (population_a + population_b + population_c),
                label=labels[i], ls='-', alpha=0.5)

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.xlabel('Number of Seeds')
    plt.ylabel('Fraction of Influenced Nodes (F(S))')
    plt.savefig(filename + '_total_influenced.png', bbox_inches='tight')
    plt.close(fig)

    # comparison abs difference
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    index = np.arange(1, num_seeds + 1)
    for i, (influenced_a, influenced_b, influenced_c) in enumerate(zip(influenced_a_list, influenced_b_list, influenced_b_list)):
        # ax.plot(index, np.abs(np.asarray(influenced_a) / population_a - np.asarray(influenced_b) / population_b),
        #         label=labels[i], ls='-', alpha=0.5)
        ax.plot(index, np.abs(np.asarray(influenced_a) / population_a - np.asarray(influenced_b) / population_b)+
                np.abs(np.asarray(influenced_a) / population_a - np.asarray(influenced_c) / population_b)+
                np.abs(np.asarray(influenced_b) / population_a - np.asarray(influenced_c) / population_b),
                label=labels[i], ls='-', alpha=0.5)

    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.xlabel('Number of Seeds')
    plt.ylabel('Absolute difference of Influenced Nodes \n (|Fa - Fb| + |Fa - Fc| + |Fb - Fc| + )')
    plt.savefig(filename + '_difference_total_influenced.png', bbox_inches='tight')
    plt.close(fig)


def load_random_graph(filename, n, p, w):
    # return get_random_graph(filename+'.txt', n,p,w)
    try:
        f = open(filename + '.txt', 'r')
        G = nx.Graph()
        n, m = map(int, f.readline().split())
        print("loaded: " + filename)
        for i, line in enumerate(f):
            if i < n:
                node_str = line.split()
                u = int(node_str[0])
                color = node_str[1]
                G.add_node(u, color=color)
            else:
                edges_str = line.split()
                u = int(edges_str[0])
                v = int(edges_str[1])
                weight = float(edges_str[2])
                G.add_edge(u, v, weight=weight)
        f.close()
    except FileNotFoundError:
        print("File not found at {filename}, building new graph...")
        G = get_random_graph(filename + '.txt', n, p, w)
    return G


def save_graph(filename, G):
    if filename:
        with open(filename, 'w+') as f:
            f.write('%s %s%s' % (len(G.nodes()), len(G.edges()), os.linesep))
            for n, ndata in G.nodes(data=True):
                f.write('%s %s%s' % (n, ndata['color'], os.linesep))
            for v1, v2, edata in G.edges(data=True):
                # for it in range(edata['weight']):
                f.write('%s %s %s%s' % (v1, v2, edata['weight'], os.linesep))
        print("saved")


def get_random_graph(filename, n, p, w):
    G = nx.binomial_graph(n, p)
    color = 'blue'  # all nodes are one color
    nx.set_node_attributes(G, color, 'color')
    nx.set_edge_attributes(G, w, 'weight')

    # save_graph(filename, G)

    return G


def get_twitter_data(filename, w=None, save=False):
    '''
    reads twitter data, makes bipartition and assign group memebership
    with constant weights of infection
    '''
    f = None
    DG = None
    try:
        f = open(filename + '.txt', 'r')
        print("loaded: " + filename)
        DG = nx.DiGraph()
        n, m = map(int, f.readline().split())
        for i, line in enumerate(f):
            if i < n:
                node_str = line.split()
                u = int(node_str[0])
                color = node_str[1]
                DG.add_node(u, color=color)
            else:
                edges_str = line.split()
                u = int(edges_str[0])
                v = int(edges_str[1])
                weight = float(edges_str[2])
                if w is not None:
                    DG.add_edge(u, v, weight=w)
                else:
                    DG.add_edge(u, v, weight=weight)
        f.close()

    except FileNotFoundError:
        #
        print(" Making graph ")
        f = open('twitter/twitter_combined.txt', 'r')
        DG = nx.DiGraph()

        for line in f:
            node_a, node_b = line.split()
            DG.add_nodes_from([node_a, node_b])
            DG.add_edges_from([(node_a, node_b)])

            DG[node_a][node_b]['weight'] = w

        print("done with edges and weights ")

        G_a, G_b = community.kernighan_lin_bisection(DG.to_undirected())
        for n in G_a:
            DG.nodes[n]['color'] = 'red'
        for n in G_b:
            DG.nodes[n]['color'] = 'blue'

        save_graph(filename, DG)

    return DG


def get_facebook_data(filename, w=None, save=False):
    '''
    reads twitter data, makes bipartition and assign group memebership
    with constant weights of infection
    '''
    f = None
    G = None
    try:
        f = open(filename + '_with_communities.txt', 'r')
        print("loaded: " + filename)
        G = nx.Graph()
        n, m = map(int, f.readline().split())
        for i, line in enumerate(f):
            if i < n:
                node_str = line.split()
                u = int(node_str[0])
                color = node_str[1]
                G.add_node(u, color=color)
            else:
                edges_str = line.split()
                u = int(edges_str[0])
                v = int(edges_str[1])
                weight = float(edges_str[2])
                if w is not None:
                    G.add_edge(u, v, weight=w)
                else:
                    G.add_edge(u, v, weight=weight)
        f.close()

    except FileNotFoundError:
        #
        print(" Making graph ")

        G = facebook_circles_graph(filename, w)

        G_a, G_b = community.kernighan_lin_bisection(G.to_undirected())
        for n in G_a:
            G.nodes[n]['color'] = 'red'
        for n in G_b:
            G.nodes[n]['color'] = 'blue'

        save_graph(filename, G)

    return G
