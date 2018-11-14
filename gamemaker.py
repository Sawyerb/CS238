from donor import Donor
from election import Election

INITIAL_FUNDS = 1000000
# choose a new mean for every race
election = Election(10, .45)
donor = Donor(INITIAL_FUNDS)

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

def main(): 
	i = 0
	while(election.n_rounds != 0):
		poll = election.generatePoll()
		print("In round " + str(i) + ", candidate had " + str(round(poll, 2)) + " poll share")
		contribution = donor.makeContribution(1)
		print("\tDonor contributes " + str(contribution))
		election.updateSupport(contribution)
		i += 1

	election_result = election.runElection()
	print("\nResult: candidate received " + str(round(election_result, 2)) + " vote share")
	if(election_result > 0.5):
		won = True
		print("Candidate won")
	else:
		won = False
		print("Candidate lost")

	calculateScore(won, donor)

if __name__ == '__main__':
	main()


