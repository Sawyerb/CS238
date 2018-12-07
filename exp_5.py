"""
Experiment 5: Multiple elections

"""

import random
from scipy.stats import norm
import numpy as np
from donor import Donor
from election import Election
import continuous_solver_multi
import baseline_solver_multi
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.stats as st

POLLING_SD = 0.005

INITIAL_FUNDS = 5000

# for pomcpow
N = 10000
KA = 30 * 10
AA = 1.0/30
KO = 5 * 10
AO = 0.01
C = 200

START_SUPPORTS = [x/100.0 for x in range(0, 30, 80)]
ROUNDS = 10
MAX_WIN_REWARD = 500
MAX_LOSE_PENALTY = 0


iters = 10
NUM_ELECTIONS = 5

scores = np.zeros(shape = (iters))
contributions = np.zeros(shape = (ROUNDS, iters, NUM_ELECTIONS))
start_supports = np.random.uniform(0, 1, size = (iters, NUM_ELECTIONS))

for i in tqdm(range(iters)):
	elections = []
	for j in tqdm(range(NUM_ELECTIONS)):
		elections.append(Election(ROUNDS, start_supports[i][j], INITIAL_FUNDS))
	donor = Donor(INITIAL_FUNDS)

	score = 0
	pbar = tqdm(total = ROUNDS)
	while(elections[0].n_rounds != 0):
		polls = [e.generatePoll() for e in elections]
		
		#contribs = continuous_solver_multi.plan_pomcpow(polls, N, elections, KA, AA, KO,
		#						 AO, C, donor.funds, MAX_WIN_REWARD, MAX_LOSE_PENALTY)
		contribs = baseline_solver_multi.make_contribution(polls, donor.funds, elections)
		contributions[ROUNDS - elections[0].n_rounds][i][:] = contribs
		
		for j in range(NUM_ELECTIONS):
			donor.makeContribution(contribs[j])
			elections[j].updateSupport(contribs[j], verbose=False)
		
		pbar.update(1)

	pbar.close()
	score -= sum(contribs)
	
	for j in range(NUM_ELECTIONS):
		if(elections[j].support > 0.5):
			score += MAX_WIN_REWARD * ((ROUNDS-elections[j].n_rounds+1)/(ROUNDS+1))
		else:
			score -= MAX_LOSE_PENALTY * ((ROUNDS-elections[j].n_rounds+1)/(ROUNDS+1))

		scores[i] = score


np.save("test_multi_big_baseline_contributions.npy", contributions)
np.save('test_multi_big_baseline_scores.npy', np.array(scores))
