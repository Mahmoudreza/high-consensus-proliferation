''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 1)'''
from priorityQueue import PriorityQueue as PQ
from IC import *
import numpy as np
import multiprocessing
import utils as ut
import math
from itertools import repeat


def map_IC_timing(inp):
    G, S, v, gamma_a, gamma_b = inp
    R = 100
    priority = 0.0
    priority_a = 0.0
    priority_b = 0.0
    F_a = 0.0
    F_b = 0.0
    if v not in S:
        for j in range(R):  # run R times Random Cascade
            # for different objective change this priority selection
            T, T_a, T_b = runIC_fair_timings((G, S + [v], gamma_a, gamma_b))
            priority_a += float(T_a) / R
            priority_b += float(T_b) / R
            priority += float(T_a + T_b) / R

    return (v, priority, priority_a, priority_b)


def map_IC(G,S):
    # G, S, p = inp
    # print(S)
    return len(runIC(G, S))


def map_fair_IC(inp):
    G, S, h_l = inp
    # print(S)
    R = 500
    influenced, influenced_r, influenced_b, influenced_n = (0.0,) * 4
    pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)
    results = pool.map(runIC_fair, [(G, S,h_l) for i in range(R)])
    pool.close()
    pool.join()
    for T, T_r, T_b, T_n  in results:
        # for j in range(R):
        # T, T_a, T_b = runIC_fair(G,S)
        influenced += float(len(T)) / R
        influenced_r += float(len(T_r)) / R
        influenced_b += float(len(T_b)) / R
        influenced_n += float(len(T_n)) / R

    return (influenced, influenced_r, influenced_b, influenced_n)


def map_select_next_seed_greedy(G, S, v,h_l):
    # selects greedily
    # G, S, v = inp
    R = 100
    priority = 0.0
    if v not in S:
        for j in range(R):  # run R times Random Cascade
            # for different objective change this priority selection
            T, T_r, T_b, T_n = runIC_fair((G, S + [v],h_l))
            priority -= float(len(T)) / R

    return (v, priority)


def map_select_next_seed_log_greedy_prev(G, S, v, gamma1, gamma2):
    # selects greedily
    # G, S, v, gamma = inp
    R = 100
    priority = 0.0
    e = 1e-20
    if v not in S:
        for j in range(R):  # run R times Random Cascade
            # for different objective change this priority selection
            T, T_r, T_b, T_n = runIC_fair((G, S + [v]))
            # priority -= (math.log10(float(len(T_a)) + 1e-20) + gamma * math.log10(float(len(T_b)) + 1e-20)) / R
            priority -= (math.log10(float(len(T_r)) + 1e-20) + gamma1 * math.log10(float(len(T_b)) + 1e-20)+ gamma2 * math.log10(float(len(T_n)) + 1e-20)) / R

    return (v, priority)


def map_select_next_seed_log_greedy(G, S, v, gamma1, gamma2):
    # selects greedily
    # G, S, v, gamma = inp
    R = 100
    priority = 0.0
    e = 1e-20
    F_r = 0.0
    F_b = 0.0
    F_n = 0.0
    if v not in S:
        for j in range(R):  # run R times Random Cascade
            # for different objective change this priority selection
            T, T_r, T_b, T_n = runIC_fair((G, S + [v]))
            F_r += float(len(T_r)) / R
            F_b += float(len(T_b)) / R
            F_n += float(len(T_n)) / R
            # priority -= (math.log10(float(len(T_a)) + 1e-20) + gamma * math.log10(float(len(T_b)) + 1e-20))/R
        priority -= (math.log10(F_r + 1e-20) + gamma1 * math.log10(F_b + 1e-20) + + gamma2 * math.log10(F_n + 1e-20))

    return (v, priority)


def map_select_next_seed_root_greedy(G, S, v, gamma, beta):
    # selects greedily
    # G, S, v, gamma, beta = inp
    R = 100
    priority = 0.0
    F_r = 0.0
    F_b = 0.0
    F_n=0.0
    if v not in S:
        for j in range(R):  # run R times Random Cascade
            # for different objective change this priority selection
            T, T_r, T_b, T_n = runIC_fair((G, S + [v]))
            F_r += float(len(T_r)) / R
            F_b += float(len(T_b)) / R
            F_n += float(len(T_n)) / R

            # priority -= (float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**beta/R
        priority -= ((F_r) ** (1.0 / gamma) + (F_b) ** (1.0 / gamma)+ (F_c) ** (1.0 / gamma)) ** beta
    return (v, priority)


def map_select_next_seed_root_majority_greedy(inp):
    # selects greedily
    G, S, v, gamma = inp
    R = 100
    priority = 0.0
    F_a = 0.0
    F_b = 0.0
    if v not in S:
        for j in range(R):  # run R times Random Cascade
            # for different objective change this priority selection
            T, T_a, T_b = runIC_fair((G, S + [v]))
            F_a += float(len(T_a)) / R
            F_b += float(len(T_b)) / R

            # priority -= (float(len(T_a))**(1/gamma) + float(len(T_b))**(1/gamma))**beta/R
        priority -= ((F_a) ** (1.0 / gamma) * 0 + F_b)
    return (v, priority)


def map_select_next_seed_norm_greedy(inp):
    # selects greedily
    G, S, v, gamma = inp
    R = 100
    priority = 0.0
    if v not in S:
        for j in range(R):  # run R times Random Cascade
            # for different objective change this priority selection
            T, T_a, T_b = runIC_fair((G, S))
            priority -= ((float(len(T_a)) ** (1 / gamma) + float(len(T_b)) ** (1 / gamma)) ** gamma) / R

    return (v, priority)


# def map_select_next_seed_set_cover(G, S, v,h_l, color):
def map_select_next_seed_set_cover(inp):

    # selects greedily
    G, S, v, h_l, color = inp
    R = 100
    priority = 0.0
    priority_r = 0.0
    priority_b = 0.0
    priority_n = 0.0
    if v not in S:
        if color=='all' or G.nodes[v]['color']==color or G.nodes[v]['color']=='purple':
            for j in range(R):  # run R times Random Cascade
                # for different objective change this priority selection
                T, T_r, T_b, T_n = runIC_fair((G, S + [v],h_l))
                priority += float(len(T)) / R  # not subratacting like other formulations adding a minus later
                priority_r += float(len(T_r)) / R
                priority_b += float(len(T_b)) / R
                priority_n += float(len(T_n)) / R

    return (v, priority, priority_r, priority_b, priority_n)


def generalGreedy_parallel_inf(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    parallel computation of influence of the node, but, probably, since the computation is not that complex
    '''
    # import time
    # start = time.time()
    # define map function
    # CC_parallel(G, seed_size, .01)

    # results = []#np.asarray([])
    R = 500  # number of times to run Random Cascade
    S = []  # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):
        s = PQ()  # priority queue

        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0)  # initialize spread value
                [priority, count, task] = s.entry_finder[v]
                pool = multiprocessing.Pool(multiprocessing.cpu_count() / 2)
                results = pool.map(map_IC, [(G, S + [v], p)] * R)
                pool.close()
                pool.join()
                s.add_task(v, priority - float(np.sum(results)) / R)
                # for j in range(R): # run R times Random Cascade
                # [priority, count, task] = s.entry_finder[v]
                #  s.add_task(v, priority - float(len(runIC(G, S + [v], p)))/R) # add normalized spread value
        task, priority = s.pop_item()
        S.append(task)
        # print(i, k, time.time() - start)
    return S


def generalGreedy(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    # import time
    # start = time.time()
    R = 200  # number of times to run Random Cascade
    S = []  # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):  # cannot parallellize
        s = PQ()  # priority queue

        for v in G.nodes():
            if v not in S:
                s.add_task(v, 0)  # initialize spread value
                # [priority, count, task] = s.entry_finder[v]
                for j in range(
                        R):  # run R times Random Cascade The gain of parallelizing isn't a lot as the one runIC is not very complex maybe for huge graphs
                    [priority, count, task] = s.entry_finder[v]
                    s.add_task(v, priority - float(len(runIC(G, S + [v], p))) / R)  # add normalized spread value

        task, priority = s.pop_item()
        print(task, priority)
        S.append(task)
        # print(i, k, time.time() - start)
    return S


def generalGreedy_node_parallel(filename, G, budget, h_l, gamma1, gamma2, beta1=1.0,beta2=1.0, type_algo=1):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    # import time
    # start = time.time()
    # R = 200 # number of times to run Random Cascade
    S = []  # set of selected nodes
    influenced = []
    influenced_a = []
    influenced_b = []
    influenced_c = []
    seeds_a = []
    seeds_b = []
    seeds_c = []
    seed_range = []
    if type_algo == 1:
        filename = filename + '_greedy_'

    elif type_algo == 2:
        filename = filename + '_log_gamma_{gamma1,gamma2}_'

    elif type_algo == 3:
        filename = filename + '_root_gamma_{gamma1}_beta_{beta1,beta2}_'

    elif type_algo == 4:
        filename = filename + '_root_majority_gamma_{gamma1}_beta_{beta1,beta2}_'

    stats = ut.graph_stats(G, print_stats=False)

    try:

        influenced, influenced_a, influenced_b,influenced_c, seeds_a, seeds_b, seeds_c = ut.read_files(filename)
        S = seeds_a[-1] + seeds_b[-1]+ seeds_c[-1]

        if len(S) >= budget:
            # ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
            print(influenced_a)
            print("\n\n")
            print(influenced_b)
            print("\n\n")
            print(influenced_c)
            print(" Seed length ", len(S))

            ut.plot_influence(influenced_a, influenced_b,influenced_c, len(S), filename, stats['group_a'], stats['group_b'], stats['group_c'],
                              [len(S_a) for S_a in seeds_a], [len(S_b) for S_b in seeds_b], [len(S_c) for S_c in seeds_c])

            return (influenced, influenced_a, influenced_b,influenced_c, seeds_a, seeds_b, seeds_c)
        else:
            seed_range = range(budget - len(S))

    except FileNotFoundError:
        print('{filename} not Found ')

        seed_range = range(budget)

    # add node to S if achieves maximum propagation for current chosen + this node
    for i in seed_range:  # cannot parallellize

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # results = None
        if type_algo == 1:
            results = pool.starmap(map_select_next_seed_set_cover, zip(repeat(G), repeat(S), list(G.nodes()),repeat(h_l)))
            # results = pool.map(map_select_next_seed_greedy, ((G, S, v,h_l) for v in G.nodes()))
        elif type_algo == 2:
            results = pool.map(map_select_next_seed_log_greedy, ((G, S, v, gamma1,gamma2) for v in G.nodes()))
        elif type_algo == 3:
            results = pool.map(map_select_next_seed_root_greedy, ((G, S, v, gamma1, beta1, beta2) for v in G.nodes()))
        elif type_algo == 4:
            results = pool.map(map_select_next_seed_root_majority_greedy, ((G, S, v, gamma1) for v in G.nodes()))

        pool.close()
        pool.join()

        s = PQ()  # priority queue
        # if results == None:

        for v, priority, p_a, p_b, p_c in results:  # run R times Random Cascade The gain of parallelizing isn't a lot as the one runIC is not very complex maybe for huge graphs
            s.add_task(v, -priority)

        node, priority = s.pop_item()
        S.append(node)
        I, I_a, I_b, I_c = map_fair_IC((G, S, h_l))
        influenced.append(I)
        influenced_a.append(I_a)
        influenced_b.append(I_b)
        influenced_c.append(I_c)
        S_red = []
        S_blue = []
        S_purple = []
        group = G.nodes[node]['color']
        print(str(i + 1) + ' Selected Node is ' + str(node) + ' group ' + str(group) +' Ia = '
              + str(I_a)+' Ib = '+str(I_b) +' Ic = ' + str(I_c))
        for n in S:
            if G.nodes[n]['color'] == 'red':
                S_red.append(n)
            if G.nodes[n]['color'] == 'blue':
                S_blue.append(n)
            else:
                S_purple.append(n)

        seeds_a.append(S_red)  # id's of the seeds so the influence can be recreated
        seeds_b.append(S_blue)
        seeds_c.append(S_purple)
        # print(i, k, time.time() - start)
    # print ( "\n \n  I shouldn't be here.   ********* \n \n ")
    ut.plot_influence(influenced_a, influenced_b,influenced_c, len(S), filename, stats['group_r'], stats['group_b'],stats['group_n'],
                      [len(S_a) for S_a in seeds_a], [len(S_b) for S_b in seeds_b], [len(S_c) for S_c in seeds_c])

    ut.write_files(filename, influenced, influenced_a, influenced_b,influenced_c, seeds_a, seeds_b, seeds_c)

    return (influenced, influenced_a, influenced_b,influenced_c, seeds_a, seeds_b, seeds_c)


def generalGreedy_node_set_cover(filename, G, budget,h_l=0,color='all', seed_size_budget=14, gamma_a=1e-2, gamma_b=0, type_algo=1):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- fraction of population needs to be influenced in all three groups
    p -- propagation probability
    Output: S -- initial set of k nodes to propagate
    '''
    # import time
    # start = time.time()
    # R = 200 # number of times to run Random Cascade

    stats = ut.graph_stats(G, print_stats=False)

    if type_algo == 1:
        filename = filename + '_set_cover_reach_' + str(budget)
    elif type_algo == 2:
        filename = filename + '_set_cover_timings_reach_{budget}_gamma_a_{gamma_a}_gamma_b_{gamma_b}_'
    elif type_algo == 3:
        filename = filename + '_set_cover_timings_reach_{budget}_gamma_a_{gamma_a}_gamma_b_{gamma_a}_'

    reach = 0.0
    S = []  # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    influenced = []
    influenced_r = []
    influenced_b = []
    influenced_n = []
    seeds_r = []
    seeds_b = []
    seeds_n = []

    # try:
    #
    #     influenced, influenced_r, influenced_b, influenced_n, seeds_r, seeds_b, seeds_n = ut.read_files(filename)
    #     reach = min(influenced_r[-1] / stats['group_r'], budget) + min(influenced_b[-1] / stats['group_b'])+ min(influenced_n[-1] / stats['group_r'], budget)
    #     S = seeds_r[-1] + seeds_b[-1]+ seeds_n[-1]
    #     if reach >= budget:
    #         # ut.write_files(filename,influenced, influenced_a, influenced_b, seeds_a, seeds_b)
    #         print(influenced_r)
    #         print("\n\n")
    #         print(influenced_b)
    #         print("\n\n")
    #         print(influenced_n)
    #         print(f" reach: {reach}")
    #         ut.plot_influence(influenced_r, influenced_b, influenced_n, len(S), filename, stats['group_a'], stats['group_b'], stats['group_c'],
    #                           [len(S_a) for S_a in seeds_r], [len(S_b) for S_b in seeds_b], [len(S_c) for S_c in seeds_n])
    #         return (influenced, influenced_r, influenced_b, influenced_n, seeds_r, seeds_b, seeds_n)
    #
    # except FileNotFoundError:
    #     print(f'{filename} not Found ')

    i = 0
    S=[]
    while reach < 3 * budget:
    # while len(S) < seed_size_budget:  # cannot parallellize

        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        # pool = multiprocessing.Pool(1)

        # for v in G.nodes():
        #     results = pool.map(map_select_next_seed_set_cover, (G, S, v))

        if type_algo == 1:
            # results = pool.map(map_select_next_seed_set_cover, ((G, S, v) for v in G.nodes()))
            # results = pool.starmap(map_select_next_seed_set_cover, zip(repeat(G), repeat(S), list(G.nodes()),repeat(h_l), repeat(color)))
            results = pool.map(map_select_next_seed_set_cover, ((G, S, v,h_l, color) for v in G.nodes()))
        elif type_algo == 2:
            results = pool.map(map_IC_timing, ((G, S, v, gamma_a, gamma_b) for v in G.nodes()))
        elif type_algo == 3:
            results = pool.map(map_IC_timing, ((G, S, v, gamma_a, gamma_a) for v in G.nodes()))

        pool.close()
        pool.join()

        s = PQ()  # priority queue
        for v, p, p_a, p_b, p_c in results:  #
            # s.add_task(v, -(min(p_a / stats['group_r'], budget) + min(p_b / stats['group_b'], budget)))
            s.add_task(v, -(min(p_a / stats['group_r'], budget) + min(p_b / stats['group_b'], budget)+ min(p_b / stats['group_n'], budget)))

        node, priority = s.pop_item()
        # priority = -priority # as the current priority is negative fraction
        S.append(node)

        # results = map_select_next_seed_set_cover, ((G, S, v) for v in G.nodes())


        I, I_a, I_b, I_c = map_fair_IC((G, S,h_l))
        influenced.append(I)
        influenced_r.append(I_a)
        influenced_b.append(I_b)
        influenced_n.append(I_c)
        S_red = []
        S_blue = []
        S_purple = []
        group = G.nodes[node]['color']

        for n in S:
            if G.nodes[n]['color'] == 'red':
                S_red.append(n)
            elif G.nodes[n]['color'] == 'blue':
                S_blue.append(n)
            else:
                S_purple.append(n)

        seeds_r.append(S_red)  # id's of the seeds so the influence can be recreated
        seeds_b.append(S_blue)
        seeds_n.append(S_purple)

        # reach += -priority both are fine
        reach_a = I_a / stats['group_r']
        reach_b = I_b / stats['group_b']
        reach_c = I_c / stats['group_n']
        reach = (min(reach_a, budget) + min(reach_b, budget) + min(reach_c, budget))

        print(str(i + 1) + ' Node ID ' + str(node) +' group ' + str(group) + ' Ia  = '+str(I_a)+ ' Ib '+
              str(I_b) + ' Ic ' + str(I_c) +' each: '+ str(reach) + ' reach_a ' +  str(reach_a) +' reach_b '
              +  str(reach_b) +' reach_c ' +  str(reach_c))
        # print(i, k, time.time() - start)
        i += 1

    # ut.plot_influence(influenced_r, influenced_b, influenced_n, len(S), filename, stats['group_r'], stats['group_b'], stats['group_n'],
    #                   [len(S_r) for S_r in seeds_r], [len(S_b) for S_b in seeds_b], [len(S_n) for S_n in seeds_n])

    # ut.plot_influence_diff(influenced_r, influenced_b, influenced_n, len(S), ['Rep','Dem','Neut'], filename,
    #                     stats['group_r'], stats['group_b'], stats['group_n'])

    ut.write_files(filename, influenced, influenced_r, influenced_b, influenced_n, seeds_r, seeds_b, seeds_n)

    return (influenced, influenced_r, influenced_b, influenced_n, seeds_r, seeds_b, seeds_n)



