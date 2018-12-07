"""
Demo
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

# for pomcpow
N = 10000
KA = 30 * 10
AA = 1.0/30
KO = 5 * 10
AO = 0.01
C = 200

models = ["baseline", "pomcpow", "manual"]

while True:
	num_election = int(input("How many elections? "))
	if(num_election <= 0):
		print("Choose at least one election.")
	else:
		break

while True :
	rounds = int(input("How many round? "))
	if(rounds <= 0):
		print("Choose at least 1 round.")
	else:
		break

while True:
	model = input("Select Model (baseline/pomcpow/manual): ")
	if(model not in models):
		print("Select a valid model.")
	else:
		break

while True:
	win_reward = int(input("Max win reward? "))
	if(win_reward < 0):
		print("Choose a non-negative reward.")
	else:
		break

while True:
	lose_penalty = int(input("Max lose penalty? "))
	if(lose_penalty < 0):
		print("Choose a non-negative penalty.")
	else:
		break

while True:
	buget = int(input("Buget? "))
	if(buget < 0):
		print("Choose a non-negative buget.")
	else:
		break

print("\n")
elections = []
for j in range(num_election):
	elections.append(Election(rounds, np.random.uniform(0, 1), buget))
	print("Election " + str(j) + ": starting support " + str(elections[-1].support) + " | T : " + str(elections[-1].transition_effect))
donor = Donor(buget)

score = 0
while(elections[0].n_rounds != 0):
	print("\n")
	print("Round " + str(rounds - elections[0].n_rounds + 1))
	polls = [e.generatePoll() for e in elections]
	for i in range(num_election):
		print("Election " + str(i) + ": poll -" + str(polls[i]))

	if(model == 'baseline'):
		contribs = baseline_solver_multi.make_contribution(polls, donor.funds, elections)
	elif(model == 'pomcpow'):
		contribs = continuous_solver_multi.plan_pomcpow(polls, N, elections, KA, AA, KO,
								 AO, C, donor.funds, win_reward, lose_penalty)
	else:
		contribs = []
		b = donor.funds
		for i in range(num_election):
			print("Buget: " + str(b))
			while True:
				x = int(input("How much do you want to contribute to election " + str(i) + ": "))
				if x > b:
					print("You can't contribute that much!")
				else:
					contribs.append(x)
					b -= x
					break

	for j in range(num_election):
		if(model != 'manual'):
			print("Donor contributed " + str(contribs[j]) + " to election " + str(j))
		donor.makeContribution(contribs[j])
		elections[j].updateSupport(contribs[j], verbose=False)

	score -= sum(contribs)	
	for j in range(num_election):
		if(elections[j].support > 0.5):
			score += win_reward * ((rounds-elections[j].n_rounds+1)/(rounds+1))
		else:
			score -= lose_penalty * ((rounds-elections[j].n_rounds+1)/(rounds+1))

for i in range(num_election):
	if(elections[i].support > 0.5):
		print("Election " + str(i) + ": support was " + str(elections[i].support) + " -- Victory!!!")
	else:
		print("Election " + str(i) + ": support was " + str(elections[i].support) + " -- Defeat :(")

print("\nScore: " + str(score))