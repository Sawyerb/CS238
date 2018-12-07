"""
Baseline Solver--donates enough to swing the race if doing so if possible and necessary
"""

import scipy.stats as st

def make_contribution(support, donor_funds, candadite_funds, opp_funds, real_T = False):
	if(support > 0.5):
		return 0

	if(real_T):
		sample = 0.8931337137778281
	else:
		sample = 1

	money_percent = candadite_funds / float(candadite_funds + opp_funds)
	needed_more_money_perecent = (0.5 - support)/sample
	C = money_percent + needed_more_money_perecent
	needed_contribution = (candadite_funds -candadite_funds*C - opp_funds*C)/(C-1)
	if(needed_contribution >= donor_funds or needed_contribution < 0):
		return 0
	else:
		return needed_contribution

