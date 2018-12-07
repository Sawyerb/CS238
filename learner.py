"""
DEPRECATED--basic solver using a particle filter to estimate true support

"""

import random
from scipy.stats import norm
import numpy as np
from donor import Donor
from election import Election
from gamemaker import calculateScore

POLLING_SD = 0.02
INITIAL_FUNDS = 1000000



# particle filter algorithm from page 140
def updateBelief(b, a, oldPoll, poll, numSamples, topSamples):
	samples = []
	weights = []
	for i in range(numSamples):
		moneyEffect = random.uniform(-0.000001,0.000003) # range of spending effects
		oldSupport = oldPoll 
		newSupport = moneyEffect*a + oldPoll
		weight = norm.pdf(poll, loc = newSupport, scale = POLLING_SD)
		samples.append(moneyEffect)
		weights.append(weight)

	s = sum(weights)
	weights = [w/s for w in weights]

	moneyEffect = 0
	for i in range(topSamples):
		sampleIndex = np.random.choice(numSamples, p = weights)
		sample = samples[sampleIndex]
		moneyEffect += sample

	return moneyEffect/topSamples

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

election = Election(10, .45)
donor = Donor(INITIAL_FUNDS)
belief = 1e-5# starting belief

i = 0
while(election.n_rounds != 0):
	poll = election.generatePoll()
	print("In round " + str(i) + ", candidate had " + str(round(poll, 2)) + " vote share")
	if(i != 0 and swingContribution != 0):
		belief = updateBelief(belief, swingContribution, oldPoll, poll, 100, 50)
	print("belief error: " + str(belief - 2e-6))
	swingContribution = (0.5 - poll)/belief # amont needed to win the election
	if(belief < 0 or swingContribution < 0 or swingContribution > donor.funds):
		swingContribution = 0
	print("In round " + str(i) + ", donor contributed " + str(swingContribution))
	donor.makeContribution(swingContribution)
	election.updateSupport(swingContribution)
	
	oldPoll = poll
	i += 1

election_result = election.runElection()
print("Result: candidate received " + str(round(election_result, 2)) + " vote share")
if(election_result > 0.5):
	won = True
	print("Candidate won")
else:
	won = False
	print("Candidate lost")

calculateScore(won, donor)


