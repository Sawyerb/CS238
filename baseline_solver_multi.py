"""
Generalization of baseline solver to problem with multiple races
"""

import scipy.stats as st
import numpy as np

def make_contribution(polls, donor_funds, elections, real_T = False):
	order = np.arange(len(elections))
	np.random.shuffle(order)
	contributions = np.zeros(shape = len(elections))

	for i in order:
		if(polls[i] > 0.5):
			contributions[i] = 0
			continue

		if(real_T):
			sample = 0.8931337137778281
		else:
			sample = 1

		money_percent = elections[i].money / float(elections[i].money + elections[i].opp_money)
		needed_more_money_perecent = (0.5 - polls[i])/sample
		C = money_percent + needed_more_money_perecent
		needed_contribution = (elections[i].money - elections[i].money*C - elections[i].opp_money*C)/(C-1)

		if(needed_contribution >= donor_funds or needed_contribution < 0):
			contributions[i] = 0
		else:
			contributions[i] = needed_contribution
			donor_funds -= needed_contribution

	return contributions
