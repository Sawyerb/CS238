import random
from scipy.stats import norm
import numpy as np
from donor import Donor
from election import Election
from gamemaker import calculateScore
import continuous_solver

POLLING_SD = 0.02
INITIAL_FUNDS = 1000000

N = 1000
KA = 30
AA = 1.0/30
KO = 5
AO = 0.01
C = 110

START_SUPPORT = 0.45
ROUNDS = 10

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

election = Election(ROUNDS, START_SUPPORT)
donor = Donor(INITIAL_FUNDS)
belief = 1e-5# starting belief

i = 0
score = 0
contributed = 0
while(election.n_rounds != 0):
	poll = election.generatePoll()
	print("In round " + str(i) + ", candidate had " + str(round(poll, 2)) + " vote share")
	contribution = continuous_solver.plan_pomcpow(poll, election.n_rounds, N, KA, AA, KO,
							 AO, C, START_SUPPORT, donor.funds, election.n_rounds)


	print("In round " + str(i) + ", donor contributed " + str(contribution))
	donor.makeContribution(contribution)
	election.updateSupport(contribution)
	
	score -= contribution
	contributed += contribution
	if(election.support > 0.5):
		score += 100 * ((ROUNDS-election.n_rounds+1)/ROUNDS+1)

	i += 1
	#x = 1/0

election_result = election.runElection()
print("Result: candidate received " + str(round(election_result, 2)) + " vote share")
if(election_result > 0.5):
	won = True
	print("Candidate won")
else:
	won = False
	print("Candidate lost")

print(INITIAL_FUNDS - contributed)

print("Score: " + str(score))

#calculateScore(won, donor)


