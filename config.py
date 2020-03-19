import numpy as np

class infMaxConfig(object):
	def __init__(self):
		self.synthetic1  =  True
		self.synthetic2  =  False
		self.timing_test =  False
		self.twitter     =  False
		self.facebook    =  False

		if self.synthetic1:


			self.num_nodes = 1000
			# self.color='all'
			self.color='red'
			# self.color='blue'
			# self.color='purple'
			self.h_l=1
			self.p_with = .025

			self.p_acrosses = [ 0.001, 0.025, 0.015, 0.005] # experiments for dataset params

			self.p_across =.001

			self.group_ratios = [0.5,0.55, 0.6, 0.65,0.7]  # experiments for dataset params

			self.group_ratio = 0.7

			self.gammas_log = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

			self.gamma_log = 1.0

			self.gammas_root = [2.0,3.0,4.0,5.0,6.0,8.0,10.,15,30]

			self.gammas_root_majority = [1.1, 1.2, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 10.0]

			self.beta_root = [1.0]#,2.0,3.0, 4.0 ]

			self.gamma_root = 2.0

			self.seed_size = 12

			self.types = [1,2]

			self.type = 2

			self.filename = 'results/synthetic_data_h_l_'+str(self.h_l)+'_'
			# self.filename_r = 'graphs/'

			self.reach_list = [0.1] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

			self.gamma_timings_a_list = [1.0,0.9,0.8,0.7,0.6]

			self.gamma_timings_b_list = [0.0]
		elif self.synthetic2:

			self.num_nodes = 1000
			self.h_l=0
			self.p_with = .025

			self.p_acrosses = [ 0.001, 0.025, 0.015, 0.005] # experiments for dataset params

			self.p_across =.001

			self.group_ratios = [0.5,0.55, 0.6, 0.65,0.7]  # experiments for dataset params

			self.group_ratio = 0.7

			self.gammas_log = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5]

			self.gamma_log = 1.0

			self.gammas_root = [2.0,3.0,4.0,5.0,6.0,8.0,10.,15,30]

			self.gammas_root_majority = [1.1, 1.2, 1.5, 2.0, 2.5, 3.5, 4.5, 5.5, 10.0]

			self.beta_root = [1.0]#,2.0,3.0, 4.0 ]

			self.gamma_root = 2.0

			self.seed_size = 12

			self.types = [1,2]

			self.type = 2

			self.filename = 'results/synthetic_data_h_l_'+str(self.h_l)+'_'
			# self.filename_r = 'graphs/'

			self.reach_list = [0.1] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

			self.gamma_timings_a_list = [1.0,0.9,0.8,0.7,0.6]

			self.gamma_timings_b_list = [0.0]
		elif self.timing_test:

			self.filename = 'results/timing_test'
			self.num_nodes = 10
			self.p_edges = 0.5
			self.weight = .2
			self.gamma_a = 0.5

		elif self.twitter:

			self.weight = 0.1
			self.filename = 'twitter/twitter_combined'

		elif self.facebook:

			self.weight = 0.3
			self.filename = 'facebook/facebook_combined'





