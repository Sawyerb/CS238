class Donor(object):
	"""
	Donor = the main agent.
	"""
	def __init__(self, funds):
		self.funds = funds
	
	def makeContribution(self, contribution):
		if(self.funds >= contribution):
			self.funds -= contribution
			return contribution
		else:
			return 0