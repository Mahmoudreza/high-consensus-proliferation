''' File for testing different files in parallel
'''

from config import infMaxConfig
import networkx as nx
from IC import runIC, avgSize, runIC_fair_timings
# from CCparallel import CC_parallel
from generalGreedy import *
import multiprocessing
from heapq import nlargest
from GenerateGraph import generateGraphNPP
import utils as ut
# import matplotlib.pylab as plt
import os
# from load_facebook_graph import *
# import matplotlib.pyplot as plt
from copy import deepcopy
import GenerateGraph

class fairInfMaximization(infMaxConfig):

    def __init__(self, num=-1):
        super(fairInfMaximization, self).__init__()

        if self.synthetic1:
            n=self.num_nodes
            color = self.color
            # n=10000
            # filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}'
            # print(self.p_with, self.p_across)
            # self.G = ut.load_graph(filename, self.p_with, self.p_across, self.group_ratio, self.num_nodes)
            filename_nodes = 'graphs/nodes_'+str(n)
            filename_edges = 'graphs/edges_'+str(n)
            # self.h_l=0
            self.node_clr, self.G = GenerateGraph.read_graph(filename_nodes, filename_edges)
        elif self.synthetic2:
                n = self.num_nodes
                # n=10000
                # filename = f'{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}'
                # print(self.p_with, self.p_across)
                # self.G = ut.load_graph(filename, self.p_with, self.p_across, self.group_ratio, self.num_nodes)
                filename_nodes = 'graphs/nodes_' + str(n)
                filename_edges = 'graphs/edges_' + str(n)
                # self.h_l = 1
                self.node_clr, self.G = GenerateGraph.read_graph(filename_nodes, filename_edges)
        elif self.timing_test:
            # filename = f'{self.filename}_{self.num_nodes}_{self.p_edges}_{self.weight}'
            filename = self.filename + '_violates_{num}_'
            self.G = ut.load_random_graph(filename, self.num_nodes, self.p_edges, self.weight)

        elif self.twitter:
            # filename = self.filename+f'_with_communities'
            self.G = ut.get_twitter_data(self.filename, w=self.weight, save=True)

        elif self.facebook:
            # filename = self.filename+f'_with_communities'
            # G = #ut.get_facebook_data(self.filename, w = self.weight, save = True)
            self.G = facebook_circles_graph(self.filename, self.weight, save_double_edges=True)

        # self.stats = ut.graph_stats(self.G)

    def test_timing_formulation(self, num):
        runs = 1000

        X = [0, 1]
        Y = [0, 1, 2]
        # for i in range(2):
        #     X.append(random.randInt(0,2))
        # Y = deep_copy(X)
        # for in range(2):
        #     X.append(random.randInt(4,))
        v = [6]

        inf_R_x = 0
        inf_R_y = 0
        inf_x = 0
        inf_y = 0
        for i in range(runs):
            T_1, T_a, T_b = runIC_fair_timings((self.G, X + v, self.gamma_a, self.gamma_a))
            T_2, T_a, T_b = runIC_fair_timings((self.G, X, self.gamma_a, self.gamma_a))
            inf_x += T_2
            inf_R_x += (T_1 - T_2)
            T_1, T_a, T_b = runIC_fair_timings((self.G, Y + v, self.gamma_a, self.gamma_a))

            T_2, T_a, T_b = runIC_fair_timings((self.G, Y, self.gamma_a, self.gamma_a))
            inf_y += T_2
            inf_R_y += (T_1 - T_2)

        print(' X : {inf_x / runs}, diff: {inf_R_x / runs}  Y: {inf_y / runs} diff: {inf_R_y / runs}')

        if inf_R_x / runs < inf_R_y / runs:
            ut.save_graph(self.filename + '_violates_{num}_.txt', self.G)
            nx.draw(self.G)
            plt.savefig("graph" + '_violates_{num}_.png')
            print(" \n \n ********** VIOLATED *********** \n\n\n")

    def run_root_formulation(self):

        filename = str(self.filename)+'_'+str(self.num_nodes)+'_'+str(self.p_with)+'_'+str(self.p_across)+'_'+str(self.group_ratio)
        influenced_a_list = []
        influenced_b_list = []
        labels = []
        for gamma in self.gammas_root:

            for beta in self.beta_root:
                influenced, influenced_a, influenced_b, seeds_a, seeds_b = generalGreedy_node_parallel(filename, self.G,
                                                                                                       self.seed_size,
                                                                                                       gamma, beta=beta,
                                                                                                       type_algo=3)

                # ut.plot_influence(influenced_a, influenced_b, self.seed_size , filename , self.stats['group_a'], self.stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
                influenced_a_list.append(influenced_a)
                influenced_b_list.append(influenced_b)
            labels.append('gamma = {gamma}')
        influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename)
        influenced_a_list.append(influenced_a)
        influenced_b_list.append(influenced_b)
        labels.append("Greedy")
        filename = "results/comparison_root_"
        ut.plot_influence_diff(influenced_a_list, influenced_b_list, self.seed_size, labels, filename,
                               self.stats['group_a'], self.stats['group_b'])

    def run_root_majority_formulation(self):
        '''
        root of majority group alone
        '''
        filename = '{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}'
        influenced_a_list = []
        influenced_b_list = []
        labels = []
        for gamma in self.gammas_root_majority:

            for beta in self.beta_root:
                influenced, influenced_a, influenced_b, seeds_a, seeds_b = generalGreedy_node_parallel(filename, self.G,
                                                                                                       self.seed_size,
                                                                                                       gamma, beta=beta,
                                                                                                       type_algo=4)

                # ut.plot_influence(influenced_a, influenced_b, self.seed_size , filename , self.stats['group_a'], self.stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
                influenced_a_list.append(influenced_a)
                influenced_b_list.append(influenced_b)
            labels.append('gamma = {gamma}')
        influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename)
        influenced_a_list.append(influenced_a)
        influenced_b_list.append(influenced_b)
        labels.append("Greedy")
        filename = "results/comparison_root_majority_"
        ut.plot_influence_diff(influenced_a_list, influenced_b_list, self.seed_size, labels, filename,
                               self.stats['group_a'], self.stats['group_b'])

    def run_log_formulation(self):

        filename = '{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}'
        influenced_a_list = []
        influenced_b_list = []
        labels = []
        for gamma in self.gammas_log:
            influenced, influenced_a, influenced_b, seeds_a, seeds_b = generalGreedy_node_parallel(filename, self.G,
                                                                                                   self.seed_size,
                                                                                                   gamma, type_algo=2)

            # ut.plot_influence(influenced_a, influenced_b, self.seed_size , filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
            influenced_a_list.append(influenced_a)
            influenced_b_list.append(influenced_b)
            labels.append('gamma = {gamma}')
        influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename)
        influenced_a_list.append(influenced_a)
        influenced_b_list.append(influenced_b)
        labels.append("Greedy")
        filename = "results/comparison_log_"
        ut.plot_influence_diff(influenced_a_list, influenced_b_list, self.seed_size, labels, filename,
                               self.stats['group_a'], self.stats['group_b'])

    def run_set_cover_formulation(self):

        filename = '{self.filename}_{self.num_nodes}'#_{self.p_with}_{self.p_across}_{self.group_ratio}'
        influenced_r_list = []
        influenced_b_list = []
        influenced_n_list = []

        for reach in self.reach_list:
            influenced, influenced_r, influenced_b,influenced_n, seeds_a, seeds_b, seeds_n = generalGreedy_node_set_cover(filename, self.G,
                                                                                                    reach, self.h_l,self.color, self.seed_size,type_algo=1)

            # ut.plot_influence(influenced_a, influenced_b, self.seed_size , filename , stats['group_a'], stats['group_b'], [len(S_a) for S_a in seeds_a] , [len(S_b) for S_b in seeds_b])
            influenced_r_list.append(influenced_r)
            influenced_b_list.append(influenced_b)
            influenced_n_list.append(influenced_n)

    def run_set_cover_timings_formulation(self):
        '''
        discounted only group a
        '''
        filename = '{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}'
        influenced_a_list = []
        influenced_b_list = []

        for reach in self.reach_list:
            for gamma_timings_a in self.gamma_timings_a_list:
                for gamma_timings_b in self.gamma_timings_b_list:
                    influenced, influenced_a, influenced_b, seeds_a, seeds_b = generalGreedy_node_set_cover(filename,
                                                                                                            self.G,
                                                                                                            reach,
                                                                                                            gamma_timings_a,
                                                                                                            gamma_timings_b,
                                                                                                            type_algo=2)

    def run_set_cover_timings_formulation_sym(self):
        '''
        discounted both the groups equally
        '''
        filename = '{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}'
        influenced_a_list = []
        influenced_b_list = []

        for reach in self.reach_list:
            for gamma_timings_a in self.gamma_timings_a_list:
                for gamma_timings_b in self.gamma_timings_b_list:
                    influenced, influenced_a, influenced_b, seeds_a, seeds_b = generalGreedy_node_set_cover(filename,
                                                                                                            self.G,
                                                                                                            reach,
                                                                                                            gamma_timings_a,
                                                                                                            gamma_timings_b,
                                                                                                            type_algo=3)

    def compare_with_greedy(self):
        '''
        compares greedy with log [with different gammas]
        and root with different gammas
        '''
        influenced_a_list = []
        influenced_b_list = []
        labels = []
        filename = '{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{self.group_ratio}'
        # self.G = ut.load_graph(filename, self.p_with, self.p_across,  group_ratio ,self.num_nodes)

        stats = ut.graph_stats(self.G, print_stats=False)
        for t in self.types:
            if t == 1:
                gammas = [1.0]  # , 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]
            elif t == 2:
                gammas = self.gammas_log
            elif t == 3:
                gammas = self.gammas_root

            for gamma in gammas:

                influenced, influenced_a, influenced_b, seeds_a, seeds_b = generalGreedy_node_parallel(filename, self.G,
                                                                                                       self.seed_size,
                                                                                                       gamma,
                                                                                                       type_algo=t)

                ut.plot_influence(influenced_a, influenced_b, self.seed_size, filename, stats['group_a'],
                                  stats['group_b'], [len(S_a) for S_a in seeds_a], [len(S_b) for S_b in seeds_b])
                influenced_a_list.append(influenced_a)
                influenced_b_list.append(influenced_b)

                if t == 1:
                    label = "Greedy"
                elif t == 2:
                    label = 'Log_gamma{gamma}'
                elif t == 3:
                    label = 'Root_gamma{gamma}'

                labels.append(label)

        filename = "results/greedy_log_root_"
        ut.plot_influence_diff(influenced_a_list, influenced_b_list, self.seed_size, labels, filename, stats['group_a'],
                               stats['group_b'])

    def calculate_greedy(self, fn, G=None):
        '''
        returns greedy algorithms's inf , inf_a , inf_b, seeds_a, seeds_b ( at each iteration and iteration being at each seed selection)
        '''
        if G == None:
            G = self.G
        # filename, G, budget, h_l, gamma1, gamma2, beta1 = 1.0, beta2 = 1.0, type_algo = 1
        return generalGreedy_node_parallel(fn, G, self.seed_size, self.h_l, self.gammas_log,self.gammas_log, type_algo=1)

    def effect_of_group_sizes(self):
        '''
        This generate the evaluation graphs for

        ii) varrying p_g_a
        '''
        influenced_a_list = []
        influenced_b_list = []
        seeds_a_list = []
        seeds_b_list = []
        seed_list = [11223344, 11224433, 33112244, 22113344]
        for group_ratio in self.group_ratios:
            # group_ratio = 0.5 #0.7
            # A loop here to run multiple times on 5 seeds
            # for seed in SEED_list:
            filename = '{self.filename}_{self.num_nodes}_{self.p_with}_{self.p_across}_{group_ratio}'

            # read in graph
            G = ut.load_graph(filename, self.p_with, self.p_across, group_ratio, self.num_nodes)

            influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename, G)

            stats = ut.graph_stats(G, print_stats=True)

            influenced_a_list.append(influenced_a)
            influenced_b_list.append(influenced_b)
            seeds_a_list.append(seeds_a)
            seeds_b_list.append(seeds_b)

        print(" ******* Finished group size analysis *******")

        return (influenced_a_list, influenced_b_list, seeds_a_list, seeds_b_list)

    def effect_of_across_group_connectivity(self):
        '''
        This generate the evaluation graphs for
        i) varrying p_across with p_g_a = 0.5

        '''
        # Have to do this for multiple runs, and or multiple graphs
        influenced_a_list = []
        influenced_b_list = []
        seeds_a_list = []
        seeds_b_list = []
        group_ratio = 0.5  # just to bring out the effect of p_across

        for p_across in self.p_acrosses:
            filename = '{self.filename}_{self.num_nodes}_{self.p_with}_{p_across}_{group_ratio}'

            # read in graph
            G = ut.load_graph(filename, self.p_with, p_across, group_ratio, self.num_nodes)

            influenced, influenced_a, influenced_b, seeds_a, seeds_b = self.calculate_greedy(filename)  #

            stats = ut.graph_stats(G, print_stats=True)

            ut.plot_influence(influenced_a, influenced_b, self.seed_size, filename, stats['group_a'], stats['group_b'],
                              [len(S_a) for S_a in seeds_a], [len(S_b) for S_b in seeds_b])

            influenced_a_list.append(influenced_a)
            influenced_b_list.append(influenced_b)
            seeds_a_list.append(seeds_a)
            seeds_b_list.append(seeds_b)

        print(" ******* Finished connectivity analysis *******")

        return (influenced_a_list, influenced_b_list, seeds_a_list, seeds_b_list)

        ## varies group sizes


if __name__ == '__main__':

    from functools import partial
    from itertools import repeat
    from multiprocessing import pool, freeze_support
    import time

    start = time.time()

    if False:
        fair_inf = fairInfMaximization()
        fair_inf.effect_of_across_group_connectivity()
        fair_inf.effect_of_group_sizes()

    if False:
        fair_inf = fairInfMaximization()
        fair_inf.run_root_formulation()

    if False:
        fair_inf = fairInfMaximization()
        fair_inf.run_root_majority_formulation()

    if False:
        fair_inf = fairInfMaximization()
        fair_inf.run_log_formulation()

    if True:
        print("hi")
        fair_inf = fairInfMaximization()
        fair_inf.run_set_cover_formulation()

    if False:
        fair_inf = fairInfMaximization()
        filename = fair_inf.filename
        fair_inf.calculate_greedy(filename)

    if False:
        fair_inf = fairInfMaximization()
        fair_inf.run_set_cover_timings_formulation()

    if False:
        fair_inf = fairInfMaximization()
        fair_inf.run_set_cover_timings_formulation_sym()

    if False:
        for i in range(15):
            fair_inf = fairInfMaximization(i)
            fair_inf.test_timing_formulation(i)

    print('Total time:', time.time() - start)

