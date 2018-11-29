
import numpy as np
import scipy.stats as st

class Election():

	def __init__(self, n_rounds, starting_support, init_money):
		self.n_rounds = n_rounds
		self.support = starting_support
		self.opp_money = init_money*(1-starting_support)
		self.money = init_money*starting_support # just pretend it's 1:1 to start
	
	def generatePoll(self):
		params = (2.6406947480849725, -0.0019424092309923147, 0.8958540043646522, 0.12025183064941297)
		# get just one sample from poll per vote distribution
		sample = st.nct.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
		poll_outcome = self.support*sample # current vote % * number of poll % per vote %
		return poll_outcome

	def updateSupport(self, new_spending):
		self.money += new_spending
		params = (0.7841752015528167, 2.1022996225901887, 0.575376571272233, 0.08088203534323207)
		# get just one sample from vote per money distribution
		sample = st.nct.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
		sample = 1
		money_percent = self.money / float(self.money + self.opp_money)
		old_support = self.support
		self.support = money_percent*sample # current money % * number of vote % per money %
		self.support = min(1, self.support) # its not possible to get more than 100% of the votes
		print("\tSupport changed from " + str(round(old_support, 2)) + " to " + str(round(self.support, 2)))
		self.n_rounds -= 1

	def runElection(self):
		return self.generatePoll()


