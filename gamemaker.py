from donor import Donor
from election import Election

INITIAL_FUNDS = 1000000 
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
	print("In round " + str(i) + ", candadite had " + str(round(poll, 2)) + " vote share")
	contribution = donor.makeContribution(poll)
	election.updateSupport(contribution)
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


