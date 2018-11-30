import random
from scipy.stats import norm
import numpy as np
from donor import Donor
from election import Election
import continuous_solver

POLLING_SD = 0.02
INITIAL_FUNDS = 1000

# for pomcpow
# N = 1000
# KA = 30
# AA = 1.0/30
# KO = 5
# AO = 0.01
# C = 110

# for pft-dpw
N = 1000
KA = 20
AA = 1.0/25
KO = 8
AO = 1.0/85
C = 50
m = 200

START_SUPPORT = 0.47
ROUNDS = 1

def calculateScore(won, donor):
	'''
	The score is a sum of how much money
	the donor has spent and whether
	they have won or not.
	'''
	losing_penalty = 0
	winning_bonus = 2000000
	score = donor.funds - INITIAL_FUNDS
	if(won):
		score += winning_bonus
	else:
		score -= losing_penalty
	print("Score: " + str(score))

election = Election(ROUNDS, START_SUPPORT, INITIAL_FUNDS)
donor = Donor(INITIAL_FUNDS)
belief = 1e-5# starting belief

i = 0
score = 0
contributed = 0
while(election.n_rounds != 0):
	poll = election.generatePoll()
	poll = 0.49
	print("In round " + str(i) + ", candidate had " + str(round(poll, 2)) + " vote share")
	contribution = continuous_solver.plan_pftdpw(poll, N, election.n_rounds, KA, AA, KO,
							 AO, C, START_SUPPORT, donor.funds, election.n_rounds, m)


	print("In round " + str(i) + ", donor contributed " + str(contribution))
	donor.makeContribution(contribution)
	election.updateSupport(contribution)
	
	score -= contribution
	print(score)
	contributed += contribution
	if(election.support > 0.5):
		score += 100 * ((ROUNDS-election.n_rounds+1)/(ROUNDS+1))
		print(100 * ((ROUNDS-election.n_rounds+1)/(ROUNDS+1)))
	print(score)

	i += 1
	#x = 1/0

print("Result: candidate received " + str(round(election.support, 2)) + " vote share")
if(election.support > 0.5):
	won = True
	print("Candidate won")
else:
	won = False
	print("Candidate lost")

print(INITIAL_FUNDS - contributed)

print("Score: " + str(score))

#calculateScore(won, donor)


