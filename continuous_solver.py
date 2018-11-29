import numpy as np
import math
import election
import donor
from scipy.stats import norm
import scipy.stats as st

POLLING_SD = 0.02

class state():
	def __init__(self, support = -1, remaining_funds = 0, candadite_funds = 0,
			 opp_funds = 0, max_rounds = 10, n_rounds = 10):
		self.support = support
		self.remaining_funds = remaining_funds
		self.candadite_funds = candadite_funds
		self.opp_funds = opp_funds
		self.max_rounds = max_rounds
		self.n_rounds = n_rounds

def plan_pomcpow(b, n, d=1, ka=1, aa=1, c=1):
	start_h = {"children": {}, "visits": 0, "seqs": {}, 
				"remaining_funds": 100}

	for i in range(n):
		s = np.random.normal(b, POLLING_SD)
		start_state = state(0.5, 1000, 0, 100, 10, 10)
		simulate_pomcpow(start_state, start_h, d, ka, aa, c)

	best_a = None
	best_val = None
	for a in start_h["children"].keys():
		stats = start_h["children"][a]
		val = stats["Q"]
		if(best_val == None or val > best_val):
			best_val = val
			best_a = a

	print(start_h)
	return best_a

def actionProgWiden(h, ka, aa, c):
	if (len(h["children"].keys()) <= ka*(h["visits"]**aa)):
		while(True):
			a = np.random.uniform(0, h["remaining_funds"])
			if(a not in h["children"].keys()):
				break
			h["children"][a] = {Q: 0, visits: 1}

	best_a = None
	best_val = None
	for a in h["children"].keys():
		stats = h["children"][a]
		val = stats["Q"] + c*math.sqrt(math.log(h["visits"])/stats["visits"])
		if(best_val == None or val > best_val):
			best_val = val
			best_a = a

	return a

def simulate_pomcpow(s, h, d, ka, aa, c):
	if(d == 0):
		return 0

	a = actionProgWiden(h, ka, aa, c)
	new_s, o, r = nextState(s, a)

	if(a not in h["children"]):
		h["children"][a] = {"children":{}, "Q":0, "visits": 1}

	if(len(h["children"][a]["children"]) <= ka*(h["children"][a]["visits"]**aa)):
		if((a, o) not in h["seqs"]):
			h["seqs"][(a, o)] = 0
		h["seqs"][(a,o)] += 1
	else:
		o = np.random.choice((seq[1] for seq in h["seqs"].keys()), 
				v/sum(h["seqs"].values() for v in h["seqs"].values()))



	if(o not in h["children"][a]["children"]):
		h["children"][a]["children"][o] = {"children": {}, "visits": 0, "seqs": {}, 
											"states": [], "weigts": {}, 
											"remaining_funds": h["remaining_funds"] - a}

		total = r + rollout(new_s, h["children"][a]["children"][o], d-1)
	else:
		# is this right?
		h["children"][a]["children"][o]["states"].append(new_s)
		h["children"][a]["children"][o]["weigts"][new_s] = norm.pdf(o, loc = new_s.support, scale = POLLING_SD)

		new_s = np.random.choice((s for s in h["children"][a]["children"][o]["states"]), 
				v/sum(h["children"][a]["children"][o]["weigts"].values() for v in h["children"][a]["children"][o]["weigts"].values()))
		r = generate_reward(new_s, a)
		total = r + simulate_pomcpow(new_s, h["children"][a]["children"][o], d-1)

	h["visits"] += 1
	h["children"][a]["visits"] += 1
	h["children"][a]["Q"] += h["children"][a]["Q"] + (total-h["children"][a]["Q"])/h["children"][a]["visits"]

	return total


def nextState(s, a):
	params = (0.7841752015528167, 2.1022996225901887, 0.575376571272233, 0.08088203534323207)
	new_s = state()
	new_s.remaining_funds = s.remaining_funds - a
	new_s.candadite_funds = s.candadite_funds + a
	new_s.opp_funds = s.opp_funds
	new_s.max_rounds = s.max_rounds

	# get just one sample from vote per money distribution
	sample = st.nct.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
	money_percent = new_s.candadite_funds / float(new_s.candadite_funds + new_s.opp_funds)
	new_s.support = min(1, money_percent*sample) # current money % * number of vote % per money %
											     # its not possible to get more than 100% of the votes
	new_s.n_rounds = s.n_rounds - 1

	o = generate_obs(new_s)

	r = generate_reward(new_s, a)

	return (new_s, o, r)

def generate_obs(s):
	params = (2.6406947480849725, -0.0019424092309923147, 0.8958540043646522, 0.12025183064941297)
	# get just one sample from poll per vote distribution
	sample = st.nct.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
	o = s.support*sample # current vote % * number of poll % per vote %
	return o

def generate_reward(s, a):
	r = 0
	r -= a
	if(s.support > 0.5):
		r += 100 * ((s.max_rounds-s.n_rounds+1)/s.max_rounds+1)
	
	return r

def rollout(s, h, d):
	#assue that the latest poll reflects the true support and donate accordingly
	if(s.support > 0.5):
		return sum(100 * ((s.max_rounds-n+1)/s.max_rounds+1) for n in range(s.n_rounds-1, -1, -1))
	else:
		params = (0.7841752015528167, 2.1022996225901887, 0.575376571272233, 0.08088203534323207)
		sample = st.nct.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
		money_percent = s.candadite_funds / float(s.candadite_funds + s.opp_funds)
		needed_contribution = (0.5-s.support)/money_percent
		if(needed_contribution > s.remaining_funds):
			return 0
		else:
			return (-1*needed_contribution) +  sum(100 * ((s.max_rounds-n+1)/s.max_rounds+1) for n in range(s.n_rounds-2, -1, -1))



print(plan_pomcpow(0.5, 10))