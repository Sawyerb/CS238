import numpy as np
import math
import election
import donor
from scipy.stats import norm
import scipy.stats as st
from tqdm import tqdm
import random
import copy


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

def plan_pomcpow(b, n, d=1, ka=1, aa=1, ko = 1, ao = 1, c=1, start_support = 0.5, 
				 start_funds = 1, max_rounds = 10):
	start_h = {"children": {}, "visits": 1, "seqs": {}, 
				"remaining_funds": start_funds}

	for i in tqdm(range(n)):
		s = np.random.normal(b, POLLING_SD)
		start_state = state(start_support, start_funds, start_funds*start_support, start_funds*(1-start_support), max_rounds, max_rounds)
		simulate_pomcpow(start_state, start_h, d, ka, aa, ko, ao, c)

	best_a = 0
	best_val = 0
	for a in start_h["children"].keys():
		stats = start_h["children"][a]
		print(str(a) + " | " + str(stats["Q"]) + " | " + str(stats["visits"]) + " | " + str(math.sqrt(math.log(start_h["visits"])/stats["visits"])))
		val = stats["Q"]
		if(val > best_val):
			best_val = val
			best_a = a

	return best_a

def actionProgWiden(s, h, ka, aa, c):
	if (len(h["children"].keys()) <= ka*(h["visits"]**aa)):
		while(True):
			params = (-1.220215081837054, 0.9160324660574186, 0.638390131996225, 0.0798918035032058)
			# get just one sample from vote per money distribution
			sample = st.johnsonsu.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
			sample = 1
			money_percent = s.candadite_funds / float(s.candadite_funds + s.opp_funds)
			needed_more_money_perecent = (0.5 - s.support)/sample

			C = money_percent + needed_more_money_perecent
			needed_contribution = (s.candadite_funds -s.candadite_funds*C - s.opp_funds*C)/(C-1)

			needed_contribution = max(10, needed_contribution)
			max_contribution = min(h["remaining_funds"], needed_contribution*3)

			a = np.random.uniform(0, max_contribution)
			if(a not in h["children"].keys()):
				h["children"][a] = {"Q": 0, "visits": 1, "children":{}}
				break

	best_a = None
	best_val = None
	for a in h["children"].keys():
		stats = h["children"][a]
		val = stats["Q"] + c*math.sqrt(math.log(h["visits"])/stats["visits"])
		if(best_val == None or val > best_val):
			best_val = val
			best_a = a
	
	return best_a

def simulate_pomcpow(s, h, d, ka, aa, ko, ao, c):
	if(d == 0):
		return 0

	a = actionProgWiden(s, h, ka, aa, c)
	new_s, o, r = nextState(s, a)

	if(len(h["children"][a]["children"]) <= ko*(h["children"][a]["visits"]**ao)):
		if((a, o) not in h["seqs"]):
			h["seqs"][(a, o)] = 0
		h["seqs"][(a,o)] += 1
	else:
		tot = sum(h["seqs"].values())
		o = np.random.choice(a=[seq[1] for seq in h["seqs"].keys()], 
							 p=[v/tot for v in h["seqs"].values()])

	if(o not in h["children"][a]["children"]):
		h["children"][a]["children"][o] = {"children": {}, "visits": 1, "seqs": {}, 
											"states": [], "weigts": {}, 
											"remaining_funds": h["remaining_funds"] - a}

		total = r + rollout(new_s, h["children"][a]["children"][o], d-1)
	else:
		# is this right?
		h["children"][a]["children"][o]["states"].append(new_s)
		h["children"][a]["children"][o]["weigts"][new_s] = norm.pdf(o, loc = new_s.support, scale = POLLING_SD)

		tot = sum(h["children"][a]["children"][o]["weigts"].values())
		
		# is this necessary?
		if(tot == 0):
			new_s = np.random.choice([s for s in h["children"][a]["children"][o]["states"]])
		else:
			new_s = np.random.choice(a = [s for s in h["children"][a]["children"][o]["states"]], 
				    p = [v/tot for v in h["children"][a]["children"][o]["weigts"].values()])
		r = generate_reward(new_s, a)
		total = r + simulate_pomcpow(new_s, h["children"][a]["children"][o], d-1, ka, aa, ko, ao, c)

	h["visits"] += 1
	h["children"][a]["visits"] += 1
	h["children"][a]["Q"] += (total-h["children"][a]["Q"])/h["children"][a]["visits"]

	return total


def nextState(s, a):
	params = (-1.220215081837054, 0.9160324660574186, 0.638390131996225, 0.0798918035032058)
	new_s = state()
	new_s.remaining_funds = s.remaining_funds - a
	new_s.candadite_funds = s.candadite_funds + a
	new_s.opp_funds = s.opp_funds
	new_s.max_rounds = s.max_rounds

	# get just one sample from vote per money distribution
	sample = st.johnsonsu.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
	sample = 1
	money_percent = new_s.candadite_funds / float(new_s.candadite_funds + new_s.opp_funds)
	new_s.support = min(1, money_percent*sample) # current money % * number of vote % per money %
											     # its not possible to get more than 100% of the votes
	new_s.n_rounds = s.n_rounds - 1

	o = generate_obs(new_s)

	r = generate_reward(new_s, a)

	return (new_s, o, r)

def generate_obs(s):
	params = (-0.11831752848651322, 0.898170472604464, 0.06716771963319479)
	# get just one sample from poll per vote distribution
	sample = st.tukeylambda.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
	o = s.support*sample # current vote % * number of poll % per vote %
	return o

def generate_reward(s, a):
	r = 0
	r -= a
	if(s.support > 0.5):
		r += 100 * ((s.max_rounds-s.n_rounds+1)/(s.max_rounds+1))
	
	return r

def rollout(s, h, d):
	#assue that the latest poll reflects the true support and donate accordingly
	if(d==0):
		return 0

	if(s.support > 0.5):
		return sum(100 * ((s.max_rounds-n+1)/s.max_rounds+1) for n in range(s.n_rounds-1, -1, -1))
	else:
		params = (-1.220215081837054, 0.9160324660574186, 0.638390131996225, 0.0798918035032058)
		# get just one sample from vote per money distribution
		sample = st.johnsonsu.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
		sample = 1
		money_percent = s.candadite_funds / float(s.candadite_funds + s.opp_funds)
		needed_more_money_perecent = (0.5 - s.support)/sample
		C = money_percent + needed_more_money_perecent
		needed_contribution = (s.candadite_funds -s.candadite_funds*C - s.opp_funds*C)/(C-1)
		if(needed_contribution > s.remaining_funds):
			return 0
		else:
			return (-1*needed_contribution) +  sum(100 * ((s.max_rounds-n+1)/(s.max_rounds+1)) for n in range(s.n_rounds-2, -1, -1))


def plan_pftdpw(b, n, d=1, ka=1, aa=1, ko = 1, ao = 1, c=1, start_support = 0.5, 
				 start_funds = 1, max_rounds = 10, m=100):
	start_b = {"value": b, "children": {}, "visits": 1, "seqs": {}, 
				"remaining_funds": start_funds}

	for i in tqdm(range(n)):
		s = np.random.normal(b, POLLING_SD)
		start_state = state(start_support, start_funds, start_funds*start_support, start_funds*(1-start_support), max_rounds, max_rounds)
		simulate_pftdpw(start_state, start_b, d, ka, aa, ko, ao, c, m)

	best_a = 0
	best_val = 0
	for a in start_b["children"].keys():
		stats = start_b["children"][a]
		print(str(a) + " | " + str(stats["Q"]) + " | " + str(stats["visits"]) + " | " + str(math.sqrt(math.log(start_b["visits"])/stats["visits"])) + " | " + str(stats["Q"] + c*math.sqrt(math.log(start_b["visits"])/stats["visits"])))
		val = stats["Q"]
		if(val > best_val):
			best_val = val
			best_a = a

	return best_a

def simulate_pftdpw(s, b, d, ka, aa, ko, ao, c, m):
	if(d == 0):
		return 0

	a= actionProgWiden(s, b, ka, aa, c)
	new_s = copy.copy(s)
	new_s.candadite_funds += a
	new_s.n_rounds -= 1

	if(len(b["children"][a]["children"]) <= ko*(b["children"][a]["visits"]**ao)):
		new_b, r = updateBelief(b, new_s, a, m)
		new_s.support = new_b
		total = r + rollout(new_s, b["children"][a]["children"][new_b], d-1)

	else:
		key, new_b = random.choice(list(b["children"][a]["children"].items()))
		new_s.support = new_b["value"]
		r = new_b["reward"]
		total = r + simulate_pftdpw(new_s, new_b, d-1, ka, aa, ko, ao, c, m)

	b["visits"] += 1
	b["children"][a]["visits"] += 1
	b["children"][a]["Q"] += (total-b["children"][a]["Q"])/b["children"][a]["visits"]

	return total

# particle filter algorithm from page 140
def updateBelief(b, s, a, numSamples):
	samples = []
	weights = []

	for i in range(numSamples):
		rand_support = np.random.normal(s.support, POLLING_SD)
		rand_state = state(rand_support,
						   s.remaining_funds, s.remaining_funds*rand_support, 
						s.remaining_funds*(1-rand_support), 1, 1)
		new_s, o, r = nextState(rand_state, a)
		weight = norm.pdf(o, loc = rand_support, scale = POLLING_SD)

		samples.append(rand_support)
		weights.append(weight)

	tot = sum(weights)
	weights = [w/tot for w in weights]

	new_support = 0
	topSamples = int(numSamples*0.1)
	for i in range(topSamples):
		sampleIndex = np.random.choice(a = numSamples, p = weights)
		sample = samples[sampleIndex]
		new_support += sample

	new_b = new_support/topSamples
	r = -1*a
	if(new_b > 0.5):
		r += 100 * ((s.max_rounds-s.n_rounds+1)/(s.max_rounds+1))

	b["children"][a]["children"][new_b] = {"value": new_b, "reward": r, "children": {},
					"visits": 1, "seqs": {}, "remaining_funds": b["remaining_funds"] - a}
	return new_b, r

prior = 0.5
n = 1000
ka = 30
aa = 1.0/30
ko = 5
ao = 0.01
c = 110
d = 10

#print(plan_pftdpw(prior, n, d, ka, aa, ko, ao, c, start_funds=10000))