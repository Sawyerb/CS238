import scipy.stats as st

def make_contribution(support, donor_funds, candadite_funds, opp_funds):
	if(support > 0.5):
		return 0

	params = (-1.220215081837054, 0.9160324660574186, 0.638390131996225, 0.0798918035032058)
	# get just one sample from vote per money distribution
	sample = st.johnsonsu.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
	sample = 1
	
	money_percent = candadite_funds / float(candadite_funds + opp_funds)
	needed_more_money_perecent = (0.5 - support)/sample
	C = money_percent + needed_more_money_perecent
	needed_contribution = (candadite_funds -candadite_funds*C - opp_funds*C)/(C-1)
	#print("needed: " + str(needed_contribution))
	#print("remaining: " + str(s.remaining_funds))
	if(needed_contribution >= donor_funds):
		return 0
	else:
		return needed_contribution

