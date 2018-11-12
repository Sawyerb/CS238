
import numpy as np

class Election():
	POLLING_SD = 0.02 

	def __init__(self, n_rounds, spending_effect_mean, spending_effect_sd, starting_support):
		self.n_rounds = n_rounds
		self.spending_effect_mean = spending_effect_mean
		self.spending_effect_sd = spending_effect_sd
		self.support = starting_support
	
	def generatePoll(self):
		poll_outcome = np.random.normal(self.support, self.POLLING_SD)
		return poll_outcome 

	def updateSupport(self, new_spending):
		spending_effect = np.random.normal(self.spending_effect_mean, self.spending_effect_sd)
		print("Support increased by " + str(round(new_spending * spending_effect, 2)))
		self.support += (new_spending * spending_effect)
		self.support = min(1, self.support) # its not possible to get more than 100% of the votes
		self.n_rounds -= 1

	def runElection(self):
		return self.generatePoll()


