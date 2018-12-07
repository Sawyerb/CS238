"""
Generalization of POMCPOW algorithm for problem with multiple elections
"""


import numpy as np
import math
import election
import donor
from scipy.stats import norm
import scipy.stats as st
from tqdm import tqdm
import random
import copy


POLLING_SD = 0.005
MAX_WIN_REWARD = 500

class state():
	def __init__(self, elections, remaining_funds):
		self.elections = elections
		self.remaining_funds = remaining_funds

class election_state():
	def __init__(self, support = -1, candadite_funds = 0,
			 opp_funds = 0, max_rounds = 10, n_rounds = 10, donation_effect = 1):
		self.support = support
		self.candadite_funds = candadite_funds
		self.opp_funds = opp_funds
		self.max_rounds = max_rounds
		self.n_rounds = n_rounds
		self.donation_effect = donation_effect

def plan_pomcpow(polls, n, elections, ka=1, aa=1, ko = 1, ao = 1, c=1, total_funds = 1, max_win_reward = 500, max_lose_reward = 0):
	start_h = {"children": {}, "visits": 1, "seqs": {}, 
				"remaining_funds": total_funds}

	for i in tqdm(range(n)):
		elects = []
		for j in range(len(elections)):
			es = np.random.normal(polls[j], POLLING_SD)
			elects.append(election_state(es, total_funds*polls[j], total_funds*(1-polls[j]), elections[0].n_rounds, elections[0].n_rounds, elections[j].transition_effect))
		start_state = state(elects, total_funds)
		simulate_pomcpow(start_state, start_h, elections[0].n_rounds, ka, aa, ko, ao, c, max_win_reward, max_lose_reward)

	best_a = 0
	best_val = 0
	for a in start_h["children"].keys():
		stats = start_h["children"][a]
		#print(str(a) + " | " + str(stats["Q"]) + " | " + str(stats["visits"]) + " | " + str(math.sqrt(math.log(start_h["visits"])/stats["visits"])))
		val = stats["Q"]
		if(val > best_val):
			best_val = val
			best_a = a

	try:
		best_a = best_a[1:-1].split(',')
	except:
		print(best_a)
		return [0.0 for i in range(len(elections))]
	best_a = [float(a) for a in best_a]
	return best_a

def actionProgWiden(s, h, ka, aa, c):
	
	do_nothing = [0.0 for i in range(len(s.elections))]
	if(str(do_nothing) not in h["children"].keys()):
		h["children"][str(do_nothing)] = {"Q": 0, "visits": 1, "children":{}}

	if (len(h["children"].keys()) <= ka*(h["visits"]**aa)):
		while(True):
			order = np.arange(len(s.elections))
			np.random.shuffle(order)


			remaining_funds = h["remaining_funds"]

			actions = []
			for i in order:
				money_percent = s.elections[i].candadite_funds / float(s.elections[i].candadite_funds + s.elections[i].opp_funds)
				needed_more_money_perecent = (0.5 - s.elections[i].support)/s.elections[i].donation_effect

				C = money_percent + needed_more_money_perecent
				needed_contribution = (s.elections[i].candadite_funds -s.elections[i].candadite_funds*C - s.elections[i].opp_funds*C)/(C-1)

				needed_contribution = max(10, needed_contribution)
				max_contribution = min(remaining_funds, needed_contribution*1.5)
	
				contrib = np.random.uniform(0, max_contribution)
				remaining_funds -= contrib
				actions.append(contrib)

			if(str(actions) not in h["children"].keys()):
				h["children"][str(actions)] = {"Q": 0, "visits": 1, "children":{}}
				break

	best_a = None
	best_val = 0

	min_val = abs(min(a["Q"] for a in h["children"].values()))
	adj_val = [a["Q"] + min_val + 1 for a in  h["children"].values()]
	tot_val = sum(adj_val)
	if(tot_val != 0):
		p_vals = [v/tot_val for v in adj_val]
		flattner = np.mean(p_vals)
		tot_val = sum([v + flattner for v in p_vals])
		p_vals = [(v + flattner)/tot_val for v in p_vals]
	else:
		p_vals = [1/len(h["children"]) for a in range(0, len(h["children"]))]

	best_a_ind =  np.random.choice(a = range(len(p_vals)), p = p_vals)
	actions = list(h["children"].keys()) 
	best_a = actions[best_a_ind][1:-1].split(',')
	best_a = [float(a) for a in best_a]
	return best_a

def simulate_pomcpow(s, h, d, ka, aa, ko, ao, c, max_win_reward, max_lose_reward):
	if(d == 0):
		return 0

	actions = actionProgWiden(s, h, ka, aa, c)
	a = str(actions)
	new_s, o, r = nextState(s, actions, max_win_reward, max_lose_reward)
	obs = str(o)

	if(len(h["children"][a]["children"]) <= ko*(h["children"][a]["visits"]**ao)):
		if((a, obs) not in h["seqs"]):
			h["seqs"][(a, obs)] = 0
		h["seqs"][(a,obs)] += 1
	else:
		tot = sum(h["seqs"].values())
		obs = np.random.choice(a=[seq[1] for seq in h["seqs"].keys()], 
							 p=[v/tot for v in h["seqs"].values()])
		o = obs[1:-1].split(',')
		o = [float(x) for x in o]

	if(obs not in h["children"][a]["children"]):
		h["children"][a]["children"][obs] = {"children": {}, "visits": 1, "seqs": {}, 
											"states": [], "weights": {}, 
											"remaining_funds": h["remaining_funds"] - sum(actions)}

		total = r + rollout(new_s, h["children"][a]["children"][obs], d-1, max_win_reward, max_lose_reward)

	else:
		h["children"][a]["children"][obs]["states"].append(new_s)
		probs = []
		for i in range(len(new_s.elections)):
			probs.append(norm.pdf(o[i], loc = new_s.elections[i].support, scale = POLLING_SD))
		h["children"][a]["children"][obs]["weights"][new_s] = np.mean(probs)

		tot = sum(h["children"][a]["children"][obs]["weights"].values())
		
		if(tot == 0):
			new_s = np.random.choice([s for s in h["children"][a]["children"][obs]["states"]])
		else:
			new_s = np.random.choice(a = [s for s in h["children"][a]["children"][obs]["states"]], 
				    p = [v/tot for v in h["children"][a]["children"][obs]["weights"].values()])
		r = generate_reward(new_s, actions, max_win_reward, max_lose_reward)
		total = r + simulate_pomcpow(new_s, h["children"][a]["children"][obs], d-1, ka, aa, ko, ao, c, max_win_reward, max_lose_reward)

	h["visits"] += 1
	h["children"][a]["visits"] += 1
	h["children"][a]["Q"] += (total-h["children"][a]["Q"])/h["children"][a]["visits"]

	return total


def nextState(s, a, max_win_reward, max_lose_reward):
	
	elections = []
	remaining_funds = s.remaining_funds
	for i in range(len(s.elections)):
		
		remaining_funds -= a[i]
		candadite_funds = s.elections[i].candadite_funds + a[i]
		opp_funds = s.elections[i].opp_funds
		max_rounds = s.elections[i].max_rounds

		money_percent = candadite_funds / float(candadite_funds + opp_funds)
		support = min(1, money_percent*s.elections[i].donation_effect) # current money % * number of vote % per money %
												     # its not possible to get more than 100% of the votes
		n_rounds = s.elections[i].n_rounds - 1
		elections.append(election_state(support, candadite_funds, opp_funds, s.elections[i].max_rounds, n_rounds, s.elections[i].donation_effect))

	new_s = state(elections, remaining_funds)

	o = generate_obs(new_s)

	r = generate_reward(new_s, a, max_win_reward, max_lose_reward)

	return (new_s, o, r)

def generate_obs(s):
	o = []
	params = (-0.11831752848651322, 0.898170472604464, 0.06716771963319479)

	for i in range(len(s.elections)):
		# get just one sample from poll per vote distribution
		sample = st.tukeylambda.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
		o.append(s.elections[i].support*sample) # current vote % * number of poll % per vote %
	
	return o

def generate_reward(s, a, max_win_reward, max_lose_reward):
	r = 0
	r -= sum(a)
	for i in range(len(s.elections)):
		if(s.elections[i].support > 0.5):
			r += max_win_reward * ((s.elections[i].max_rounds-s.elections[i].n_rounds+1)/(s.elections[i].max_rounds+1))
		else:
			r += max_lose_reward * ((s.elections[i].max_rounds-s.elections[i].n_rounds+1)/(s.elections[i].max_rounds+1))
	
	return r

def rollout(s, h, d, max_win_reward, max_lose_reward):
	#assue that the latest poll reflects the true support and donate accordingly
	if(d==0):
		return 0

	r = 0

	order = np.arange(len(s.elections))
	np.random.shuffle(order)
	remaining_funds = s.remaining_funds

	for i in order:

		if(s.elections[i].support > 0.5):
			r += sum(max_win_reward * ((s.elections[i].max_rounds-n+1)/(s.elections[i].max_rounds+1)) for n in range(s.elections[i].n_rounds-1, -1, -1))
		else:

			money_percent = s.elections[i].candadite_funds / float(s.elections[i].candadite_funds + s.elections[i].opp_funds)
			needed_more_money_perecent = (0.5 - s.elections[i].support)/s.elections[i].donation_effect
			C = money_percent + needed_more_money_perecent
			needed_contribution = (s.elections[i].candadite_funds -s.elections[i].candadite_funds*C - s.elections[i].opp_funds*C)/(C-1)
			if(needed_contribution >= remaining_funds):
				r +=  sum(max_lose_reward * ((s.elections[i].max_rounds-n+1)/(s.elections[i].max_rounds+1)) for n in range(s.elections[i].n_rounds-2, -1, -1))
			else:
				r += -1*needed_contribution +  sum(max_win_reward * ((s.elections[i].max_rounds-n+1)/(s.elections[i].max_rounds+1)) for n in range(s.elections[i].n_rounds-2, -1, -1))
				remaining_funds -= needed_contribution

	return r