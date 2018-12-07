"""
Experiment 4: Introducing transition uncertainty
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
import scipy.stats as st

POLLING_SD = 0.005

def updateBelief(polls, money_pcts):
	T = 0
	for j in range(len(money_pcts)):
		T += polls[j+1] / money_pcts[j] 
	T /= len(money_pcts)
	return T


POLLING_SD = 0.005
INITIAL_FUNDS = 10000

# for pomcpow
N = 1000
KA = 30
AA = 1.0/30
KO = 5
AO = 0.01
C = 200

START_SUPPORTS = [x/100.0 for x in range(40, 50, 10)]
ROUNDS = 100
MAX_WIN_REWARD = 500
MAX_LOSE_PENALTY = 0

iters = 5

scores = np.zeros(shape = (len(START_SUPPORTS), iters))
contributions = np.zeros(shape = (len(START_SUPPORTS), ROUNDS, iters))
Ts = np.zeros(shape = (len(START_SUPPORTS), iters))
estimated_t = np.zeros(shape = (len(START_SUPPORTS), iters))

model = 'pomcpow'

for i in tqdm(range(len(START_SUPPORTS))):
	for j in tqdm(range(iters)):
		transition_belief = 0.75
		transition_confidence = 0.01
		max_possible = 0

		s = START_SUPPORTS[i]
		
		election = Election(ROUNDS, s, INITIAL_FUNDS)
		donor = Donor(INITIAL_FUNDS)
		Ts[i][j] = election.transition_effect

		score = 0
		money_pcts = []
		polls = []
		
		pbar = tqdm(total = ROUNDS)
		while(election.n_rounds != 0):
			poll = election.generatePoll(normal = True)

			polls.append(poll)
			if(model != 'baseline' and election.n_rounds != ROUNDS and contribution != 0):
				transition_belief = updateBelief(polls, money_pcts)
				transition_confidence = (2*ROUNDS - election.n_rounds)/(2*ROUNDS)

			estimated_t[i][j] = transition_belief

			contribution = continuous_solver.plan_pomcpow(poll, N, election.n_rounds, KA, AA, KO,
									 AO, C, election.support, donor.funds, election.money + election.opp_money,
									 election.n_rounds, MAX_WIN_REWARD, MAX_LOSE_PENALTY, transition_belief, transition_confidence)
			#contribution = baseline_solver.make_contribution(poll, donor.funds, election.money, election.opp_money, real_T = True)
			contribution += 1 # to allow us to measure the transition effect

			contributions[i][ROUNDS - election.n_rounds][j] = contribution
			donor.makeContribution(contribution)
			election.updateSupport(contribution, verbose=False, transition_effect = True)
			pbar.update(1)

			money_pcts.append(election.money / float(election.money + election.opp_money))
			score -= contribution
			
			if(election.support > 0.5):
				score += MAX_WIN_REWARD * ((ROUNDS-election.n_rounds+1)/(ROUNDS+1))
			else:
				score -= MAX_LOSE_PENALTY * ((ROUNDS-election.n_rounds+1)/(ROUNDS+1))

		pbar.close()
		scores[i][j] = score


np.save("transition_big_pomcpow_contributions.npy", contributions)
np.save('transition_big_pomcpow_scores.npy', np.array(scores))
np.save('transitoin_effects.npy', Ts)
np.save('estimated_t.npy', estimated_t)
