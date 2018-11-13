from donor import Donor
from election import Election

INITIAL_FUNDS = 1000000
# choose a new mean for every race
election = Election(10, 0.01, 0.000003, .45)
donor = Donor(INITIAL_FUNDS)

def calculateScore(won, donor):
	score = donor.funds - INITIAL_FUNDS
	if(won):
		score += 2000000
	print("Score: " + str(score))

i = 0
while(election.n_rounds != 0):
	poll = election.generatePoll()
	print("In round " + str(i) + ", candidate had " + str(round(poll, 2)) + " vote share")
	contribution = donor.makeContribution(poll)
	election.updateSupport(contribution)
	i += 1

election_result = election.runElection()
print("Result: candidate recived " + str(round(election_result, 2)) + " vote share")
if(election_result > 0.5):
	won = True
	print("Candidate won")
else:
	won = False
	print("Candidate lost")

calculateScore(won, donor)


