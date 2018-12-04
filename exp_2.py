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
N = 10000
KA = 30
AA = 1.0/30
KO = 5
AO = 0.01
C = 200

START_SUPPORTS = [x/100.0 for x in range(0, 101, 10)]
ROUNDS = 10
MAX_WIN_REWARDS = [x for x in range(100, 600, 100)]

scores = np.zeros(shape = (len(START_SUPPORTS), len(MAX_WIN_REWARDS)))
contributions = np.zeros(shape = (len(START_SUPPORTS), len(MAX_WIN_REWARDS), ROUNDS))

for i in tqdm(range(len(START_SUPPORTS))):
	for j in tqdm(range(len(MAX_WIN_REWARDS))):
		max_possible = 0


		s = START_SUPPORTS[i]
		w = MAX_WIN_REWARDS[j]
		
		election = Election(ROUNDS, s, INITIAL_FUNDS)
		donor = Donor(INITIAL_FUNDS)

		q = []
		c = []
		score = 0
		while(election.n_rounds != 0):
			poll = election.generatePoll()

			#contribution = continuous_solver.plan_pomcpow(poll, N, election.n_rounds, KA, AA, KO,
			#						 AO, C, election.support, donor.funds, election.money + election.opp_money,
			#						 election.n_rounds, w)
			contribution = baseline_solver.make_contribution(poll, donor.funds, election.money, election.opp_money)
			contributions[i][j][election.n_rounds-1] = contribution
			donor.makeContribution(contribution)
			election.updateSupport(contribution, verbose=False)
			
			score -= contribution
			
			if(election.support > 0.5):
				score += w * ((ROUNDS-election.n_rounds+1)/(ROUNDS+1))

		scores[i][j] = score

print(scores)
np.save("win_reward_big_baseline_contributions.npy", contributions)
np.save('win_reward_big_baseline_scores.npy', np.array(scores))
