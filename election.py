
import numpy as np
import scipy.stats as st

class Election():

	def __init__(self, n_rounds, starting_support, init_money):
		self.n_rounds = n_rounds
		self.support = starting_support
		self.opp_money = init_money*(1-starting_support)
		self.money = init_money*starting_support # just pretend it's 1:1 to start
	
	def generatePoll(self):
		params = (-0.11831752848651322, 0.898170472604464, 0.06716771963319479)
		# get just one sample from poll per vote distribution
		sample = st.tukeylambda.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
		poll_outcome = self.support*sample # current vote % * number of poll % per vote %
		return poll_outcome

	def updateSupport(self, new_spending, verbose = True):
		self.money += new_spending
		params = (-1.220215081837054, 0.9160324660574186, 0.638390131996225, 0.0798918035032058)
		# get just one sample from vote per money distribution
		sample = st.johnsonsu.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
		sample = 1
		money_percent = self.money / float(self.money + self.opp_money)
		old_support = self.support
		self.support = money_percent*sample # current money % * number of vote % per money %
		self.support = min(1, self.support) # its not possible to get more than 100% of the votes
		if(verbose):
			print("\tSupport changed from " + str(round(old_support, 2)) + " to " + str(round(self.support, 2)))
		self.n_rounds -= 1

	def runElection(self):
		return self.generatePoll()


