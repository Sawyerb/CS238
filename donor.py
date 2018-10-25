class Donor(object):
	"""docstring for ClassName"""
	def __init__(self, funds):
		self.funds = funds
	
	def makeContribution(self, recent_poll):
		if(self.funds > 0):
			self.funds -= 1 
			return 1
		else:
			return 0