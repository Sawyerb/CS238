"""
Experiment 1: tests of basic model

"""

import random
from scipy.stats import norm
import numpy as np
from donor import Donor
from election import Election
import continuous_solver
import baseline_solver
from matplotlib import pyplot as plt
from tqdm import tqdm

POLLING_SD = 0.005
INITIAL_FUNDS = 1000

# for pomcpow
# N = 10000
# KA = 30
# AA = 1.0/30
# KO = 5
# AO = 0.01
# C = 200

# for pft-dpw
N = 1000
KA = 20
AA = 1.0/25
KO = 8
AO = 1.0/85
C = 100
m = 500

START_SUPPORTS = [x/100.0 for x in range(0, 110, 10)]
ROUNDS = 10
MAX_WIN_REWARD = 500

iters = 50

scores = np.zeros(shape = (len(START_SUPPORTS), iters))
contributions = np.zeros(shape = (len(START_SUPPORTS), ROUNDS, iters))



for i in tqdm(range(len(START_SUPPORTS))):
	for j in tqdm(range(iters)):
		s = START_SUPPORTS[i]
		election = Election(ROUNDS, s, INITIAL_FUNDS)
		donor = Donor(INITIAL_FUNDS)
		belief = 1e-5# starting belief

		score = 0
		while(election.n_rounds != 0):
			poll = election.generatePoll()

			#contribution = continuous_solver.plan_pftdpw(poll, N, election.n_rounds, KA, AA, KO,
			# 						 AO, C, s, donor.funds, election.n_rounds, m)
			#contribution = continuous_solver.plan_pomcpow(poll, N, election.n_rounds, KA, AA, KO,
			#						 AO, C, election.support, donor.funds, election.money + election.opp_money,
			#						 election.n_rounds)
			contribution = baseline_solver.make_contribution(poll, donor.funds, election.money, election.opp_money)
			
			contributions[i][ROUNDS - election.n_rounds][j] = contribution
			donor.makeContribution(contribution)
			election.updateSupport(contribution, verbose=False)
			
			score -= contribution
			
			if(election.support > 0.5):
				score += MAX_WIN_REWARD * ((ROUNDS-election.n_rounds+1)/(ROUNDS+1))
				
		scores[i][j] = score

np.save("basic_big_baseline_contributions.npy", contributions)

np.save('basic_big_baseline_scores.npy', scores)