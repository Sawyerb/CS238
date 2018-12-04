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

iters = 5

scores = np.zeros(shape = (len(START_SUPPORTS), iters))
contributions = np.zeros(shape = (len(START_SUPPORTS), ROUNDS, iters))



for i in tqdm(range(len(START_SUPPORTS))):
	for j in range(iters):
		s = START_SUPPORTS[i]
		election = Election(ROUNDS, s, INITIAL_FUNDS)
		donor = Donor(INITIAL_FUNDS)
		belief = 1e-5# starting belief

		score = 0
		while(election.n_rounds != 0):
			poll = election.generatePoll()

			#contribution = continuous_solver.plan_pftdpw(poll, N, election.n_rounds, KA, AA, KO,
			# 						 AO, C, s, donor.funds, election.n_rounds, m)
			contribution = continuous_solver.plan_pomcpow(poll, N, election.n_rounds, KA, AA, KO,
									 AO, C, election.support, donor.funds, election.money + election.opp_money,
									 election.n_rounds)
			#contribution = baseline_solver.make_contribution(poll, donor.funds, election.money, election.opp_money)
			
			contributions[i][ROUNDS - election.n_rounds][j] = contribution
			donor.makeContribution(contribution)
			election.updateSupport(contribution, verbose=False)
			
			score -= contribution
			
			if(election.support > 0.5):
				score += MAX_WIN_REWARD * ((ROUNDS-election.n_rounds+1)/(ROUNDS+1))
				
		scores[i][j] = score

np.save("basic_big_pomcpow_contributions.npy", contributions)

np.save('basic_big_pomcpow_scores.npy', scores)


# data = scores
# plt.plot(range(0, 110, 10), data) 
# plt.xlabel('Starting Support (%)')
# plt.ylabel('Score')
# plt.savefig('basic_big_pomcpow_scores.png')

# plt.clf()

# data = contributions
# for row in data:
# 	plt.plot(range(0, 10), row)

# plt.legend(['Starting Support = ' + str(round(s, 2)) for s in np.arange(0, 1, 0.1)], loc ='upper left')
# plt.xlabel("Time Step")
# plt.ylabel("Contribution")
# plt.savefig('basic_big_comcpow_contributions.png')
