import random
from scipy.stats import norm
import numpy as np
from donor import Donor
from election import Election

POLLING_SD = 0.05
INITIAL_FUNDS = 1000000 

# particle filter algorithm from page 140
def updateBelief(b, a, o, numSamples, topSamples):
	samples = []
	weights = []
	for i in range(numSamples):
		moneyEffect = random.uniform(-0.000003,0.000003) # range of spending effects
		curSupport = random.uniform(0, 1) # range of support
		newSupport = moneyEffect*a + curSupport
		weight = norm.pdf(o, loc = newSupport, scale = POLLING_SD)
		samples.append((moneyEffect, curSupport))
		weights.append(weight)

	s = sum(weights)
	weights = [w/s for w in weights]

	moneyEffect = 0
	curSupport = 0
	for i in range(topSamples):
		sampleIndex = np.random.choice(numSamples, p = weights)
		sample = samples[sampleIndex]
		moneyEffect += sample[0]
		curSupport += sample[1]
	return [bi / topSamples for bi in (moneyEffect, curSupport)]

def calculateScore(won, donor):
	score = donor.funds - INITIAL_FUNDS
	if(won):
		score += 2000000
	print("Score: " + str(score))


election = Election(10, 0.000002, 0.0000002, .45)
donor = Donor(INITIAL_FUNDS)
belief = (0.000001, 0.5) # starting beliefs

i = 0
while(election.n_rounds != 0):
	poll = election.generatePoll()
	print("In round " + str(i) + ", candadite had " + str(round(poll, 2)) + " vote share")
	if(i != 0):
		belief = updateBelief(belief, swingContribution, poll, 10000, 1000)
	print(belief)
	swingContribution = (0.5 - belief[1])/belief[0] # amont needed to win the election
	if(belief[0] < 0 or swingContribution < 0 or swingContribution > donor.funds):
		swingContribution = 0
	print("In round " + str(i) + ", donor contributed " + str(swingContribution))
	donor.makeContribution(swingContribution)
	election.updateSupport(swingContribution)
	i += 1

election_result = election.runElection()
print("Result: candadite recived " + str(round(election_result, 2)) + " vote share")
if(election_result > 0.5):
	won = True
	print("Candidite won")
else:
	won = False
	print("Candidite lost")

calculateScore(won, donor)


